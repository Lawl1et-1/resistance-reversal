"""PertFlow: conditional flow matching in latent space with a DeepSets set encoder.

The model learns a velocity field `v_theta(z_t, t | z0, S)` transporting control cells to
the post-perturbation population. Flow matching suits this data because real perturbation
experiments give *unpaired* populations -- control cells and perturbed cells, never the same
cell before and after -- and independent-coupling CFM is defined for exactly that case.

Two design choices carry the generalisation claim:

- The perturbation set is encoded with DeepSets (sum-pooled per-gene features, then a
  readout). Sum pooling is permutation-invariant and defined for any set size, so an unseen
  *combination* is representable without ever having been observed.
- The per-gene features are the co-expression embeddings fitted on training cells only, so
  an unseen *gene* inherits behaviour from genes with similar expression profiles.

Epistemic uncertainty comes from a deep ensemble: disagreement between independently
initialised members on the predicted mean shift. `n_ensemble=1` turns the ensemble off for
the ablation, and `mode="mlp"` replaces the flow with a direct delta regressor for the
flow-vs-regressor ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import CONTROL, PerturbData
from src.data.splits import Split
from src.models.baselines import ResponseModel, gene_embeddings
from src.models.latent import LatentEncoder


def _mlp(sizes: list[int], out_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class SetEncoder(nn.Module):
    """DeepSets over perturbed-gene features. Permutation-invariant, any set size."""

    def __init__(self, feat_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.psi = _mlp([feat_dim, hidden, hidden], out_act=True)
        self.rho = _mlp([hidden, hidden, out_dim])

    def forward(self, atom_feats: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """atom_feats (B, A, F), atom_mask (B, A) with 1 for real atoms."""
        h = self.psi(atom_feats) * atom_mask.unsqueeze(-1)
        return self.rho(h.sum(dim=1))


class VelocityNet(nn.Module):
    """v(z_t, t | z0, S, context), or a direct delta regressor when mode='mlp'."""

    def __init__(
        self,
        latent_dim: int,
        feat_dim: int,
        n_contexts: int,
        hidden: int = 128,
        set_dim: int = 32,
        mode: str = "flow",
    ):
        super().__init__()
        self.mode = mode
        self.set_encoder = SetEncoder(feat_dim, hidden, set_dim)
        cond = set_dim + n_contexts
        # Flow mode takes (z_t, t) and NOT z0; see forward(). The plain regressor takes z0
        # and predicts its delta directly.
        in_dim = (latent_dim + 1 + cond) if mode == "flow" else (latent_dim + cond)
        self.net = _mlp([in_dim, hidden, hidden, latent_dim])

    def forward(self, z0, atom_feats, atom_mask, ctx_onehot, z_t=None, t=None):
        """Marginal velocity field in flow mode; direct delta regressor otherwise.

        The flow field is deliberately *not* conditioned on z0. Under independent
        coupling the training target for a given z0 is the whole perturbed marginal, so a
        z0-conditioned deterministic field can only learn E[z1 - z0 | z_t, z0] and
        integrating it collapses each starting cell onto the conditional mean -- the
        predicted population loses its spread (measured: per-dimension std 0.30 against a
        true and control value of 0.86) and the energy distance comes out worse than
        making no prediction at all. The marginal field v(z_t, t | S, context) is what
        standard conditional flow matching uses, and it transports the control marginal to
        the perturbed marginal. Per-cell structure is carried by the starting point of the
        ODE rather than by conditioning.
        """
        cond = torch.cat([self.set_encoder(atom_feats, atom_mask), ctx_onehot], dim=-1)
        parts = [z_t, t.view(-1, 1), cond] if self.mode == "flow" else [z0, cond]
        return self.net(torch.cat(parts, dim=-1))


@dataclass
class PertFlow(ResponseModel):
    """Deep ensemble of conditional flow-matching members."""

    name: str = "pertflow"
    latent_dim: int = 32
    n_ensemble: int = 5
    hidden: int = 128
    set_dim: int = 32
    emb_dim: int | None = None
    mode: str = "flow"  # "flow" | "mlp"
    steps: int = 1500
    batch_size: int = 256
    lr: float = 1e-3
    n_flow_steps: int = 20
    n_eval_cells: int = 128
    seed: int = 0
    device: str = "cpu"
    history_: list = field(default_factory=list)

    # --- fitting -----------------------------------------------------------------------

    def fit(self, data: PerturbData, split: Split, encoder: LatentEncoder) -> "PertFlow":
        self.encoder_ = encoder
        self.n_contexts_ = data.n_contexts
        self.dim_ = encoder.latent_dim_fitted_
        self.emb_ = gene_embeddings(data, split, dim=self.emb_dim)
        self.feat_dim_ = self.emb_.shape[1] + 1  # + knockdown/activation direction

        # Latent pools per (pert, context), training perturbations and contexts only.
        self.control_z_ = {
            c: encoder.transform(data.cells(CONTROL, c)) for c in split.train_contexts
        }
        groups = []
        for pert in split.train:
            if pert not in data.pert_index:
                continue
            for ctx in split.train_contexts:
                cells = data.cells(pert, ctx)
                if len(cells) and len(self.control_z_.get(ctx, [])):
                    groups.append((pert, ctx, encoder.transform(cells)))
        if not groups:
            raise ValueError("no training groups available for PertFlow")
        self.groups_ = groups

        self.members_ = []
        for m in range(self.n_ensemble):
            torch.manual_seed(self.seed * 1000 + m)
            net = VelocityNet(
                self.dim_, self.feat_dim_, self.n_contexts_, self.hidden, self.set_dim, self.mode
            ).to(self.device)
            losses = self._train_member(net, seed=self.seed * 1000 + m)
            net.eval()
            self.members_.append(net)
            self.history_.append(losses)
        return self

    def _train_member(self, net: nn.Module, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps)
        losses = []
        for step in range(self.steps):
            z0, z1, feats, mask, ctx = self._sample_batch(rng)
            if self.mode == "flow":
                t = torch.rand(len(z0), device=self.device)
                z_t = (1 - t).view(-1, 1) * z0 + t.view(-1, 1) * z1
                pred = net(None, feats, mask, ctx, z_t=z_t, t=t)
                target = z1 - z0  # independent-coupling CFM target
            else:
                pred = net(z0, feats, mask, ctx)
                target = z1 - z0
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 100 == 0:
                losses.append(float(loss.item()))
        return losses

    def _sample_batch(self, rng: np.random.Generator):
        idx = rng.integers(0, len(self.groups_), size=self.batch_size)
        z0, z1, perts, ctxs = [], [], [], []
        for i in idx:
            pert, ctx, pool = self.groups_[i]
            control = self.control_z_[ctx]
            z0.append(control[rng.integers(len(control))])
            z1.append(pool[rng.integers(len(pool))])
            perts.append(pert)
            ctxs.append(ctx)
        feats, mask = self._atom_tensors(perts)
        return (
            torch.tensor(np.array(z0), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(z1), dtype=torch.float32, device=self.device),
            feats,
            mask,
            self._ctx_tensor(ctxs),
        )

    def _atom_tensors(self, perts: list[tuple]):
        width = max(1, max(len(p) for p in perts))
        feats = np.zeros((len(perts), width, self.feat_dim_), dtype=np.float32)
        mask = np.zeros((len(perts), width), dtype=np.float32)
        for i, pert in enumerate(perts):
            for j, (gene, direction) in enumerate(pert):
                feats[i, j] = np.concatenate([self.emb_[gene], [float(direction)]])
                mask[i, j] = 1.0
        return (
            torch.tensor(feats, device=self.device),
            torch.tensor(mask, device=self.device),
        )

    def _ctx_tensor(self, ctxs: list[int]):
        onehot = np.zeros((len(ctxs), self.n_contexts_), dtype=np.float32)
        onehot[np.arange(len(ctxs)), ctxs] = 1.0
        return torch.tensor(onehot, device=self.device)

    # --- inference ---------------------------------------------------------------------

    def _z0_pool(self, context: int) -> np.ndarray:
        pool = self.control_z_.get(context)
        if pool is None or not len(pool):
            # Unseen context: fall back to the pooled training controls. This is the
            # honest behaviour -- the model has no control cells for that context -- and
            # the resulting error is what the unseen-context split is meant to expose.
            pool = np.vstack(list(self.control_z_.values()))
        return pool

    @torch.no_grad()
    def _push(self, net: nn.Module, pert: tuple, context: int, z0: np.ndarray) -> np.ndarray:
        n = len(z0)
        feats, mask = self._atom_tensors([pert] * n)
        ctx = self._ctx_tensor([min(context, self.n_contexts_ - 1)] * n)
        z0_t = torch.tensor(z0, dtype=torch.float32, device=self.device)
        if self.mode != "flow":
            return (z0_t + net(z0_t, feats, mask, ctx)).cpu().numpy()
        z = z0_t.clone()
        dt = 1.0 / self.n_flow_steps
        for i in range(self.n_flow_steps):
            t = torch.full((n,), i * dt, device=self.device)
            z = z + dt * net(None, feats, mask, ctx, z_t=z, t=t)
        return z.cpu().numpy()

    def _member_deltas(self, pert: tuple, context: int) -> np.ndarray:
        """Per-member predicted mean shift. Shape (n_ensemble, latent_dim)."""
        pool = self._z0_pool(context)
        rng = np.random.default_rng(self.seed)
        n = min(self.n_eval_cells, len(pool))
        z0 = pool[rng.choice(len(pool), n, replace=False)]
        return np.vstack([(self._push(net, pert, context, z0) - z0).mean(axis=0) for net in self.members_])

    def predict_latent_delta(self, pert: tuple, context: int) -> np.ndarray:
        return self._member_deltas(pert, context).mean(axis=0)

    def sample_latent(self, pert: tuple, context: int, z0: np.ndarray, n: int = 128) -> np.ndarray:
        """Predicted post-perturbation population, pooled across ensemble members."""
        rng = np.random.default_rng(self.seed)
        z0 = np.atleast_2d(z0)
        per = max(1, n // len(self.members_))
        out = []
        for net in self.members_:
            take = z0[rng.choice(len(z0), min(per, len(z0)), replace=len(z0) < per)]
            out.append(self._push(net, pert, context, take))
        return np.vstack(out)

    # --- uncertainty -------------------------------------------------------------------

    def epistemic_uncertainty(self, pert: tuple, context: int) -> float:
        """Ensemble disagreement on the predicted mean shift.

        Root-mean-square spread across members, averaged over latent dimensions. This is
        the raw score that conformal calibration later turns into a coverage guarantee; on
        its own it is only a relative ordering, not an error bar.
        """
        deltas = self._member_deltas(pert, context)
        if len(deltas) < 2:
            return 0.0
        return float(np.sqrt(deltas.var(axis=0, ddof=1).mean()))

    def predict_with_uncertainty(self, pert: tuple, context: int) -> tuple[np.ndarray, float]:
        deltas = self._member_deltas(pert, context)
        spread = 0.0 if len(deltas) < 2 else float(np.sqrt(deltas.var(axis=0, ddof=1).mean()))
        return deltas.mean(axis=0), spread
