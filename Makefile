PY ?= python
SMOKE := configs/smoke.yaml
FULL  := configs/full.yaml
LOGS  := logs

.PHONY: help test smoke reproduce h1 h2 h3 figures tables clean clean-results

help:
	@echo "make test       - run the test suite"
	@echo "make smoke      - tests plus every experiment on the smoke profile (CPU, ~15 min measured)"
	@echo "make reproduce  - full profile; regenerates every table and figure (hours on CPU)"
	@echo "make h1|h2|h3   - a single hypothesis on the full profile, as a background job"
	@echo "make figures    - rebuild figures from results/"
	@echo "make clean      - remove caches"

test:
	$(PY) -m pytest tests/ -q

# Smoke proves the pipeline runs end to end. Its numbers support no claim: the models are
# deliberately undertrained and the networks are small.
smoke: test
	$(PY) -m experiments.exp_h1_benchmark --config $(SMOKE)
	$(PY) -m experiments.exp_h2_calibration --config $(SMOKE)
	$(PY) -m experiments.exp_h3_search --config $(SMOKE)
	@echo "smoke profile complete -- see results/smoke/"

# Anything over ~10 minutes runs detached with a log, per the project's compute rules.
reproduce:
	@mkdir -p $(LOGS)
	$(PY) -m experiments.exp_h1_benchmark --config $(FULL) 2>&1 | tee $(LOGS)/h1_full.log
	$(PY) -m experiments.exp_h2_calibration --config $(FULL) 2>&1 | tee $(LOGS)/h2_full.log
	$(PY) -m experiments.exp_h3_search --config $(FULL) 2>&1 | tee $(LOGS)/h3_full.log
	$(MAKE) figures
	@echo "full profile complete -- see results/full/ and figures/"

h1:
	@mkdir -p $(LOGS)
	nohup $(PY) -u -m experiments.exp_h1_benchmark --config $(FULL) > $(LOGS)/h1_full.log 2>&1 &
	@echo "started; tail $(LOGS)/h1_full.log"

h2:
	@mkdir -p $(LOGS)
	nohup $(PY) -u -m experiments.exp_h2_calibration --config $(FULL) > $(LOGS)/h2_full.log 2>&1 &
	@echo "started; tail $(LOGS)/h2_full.log"

h3:
	@mkdir -p $(LOGS)
	nohup $(PY) -u -m experiments.exp_h3_search --config $(FULL) > $(LOGS)/h3_full.log 2>&1 &
	@echo "started; tail $(LOGS)/h3_full.log"

figures:
	$(PY) -m experiments.make_figures

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"

# Separate from `clean` on purpose: results are what the paper cites, so deleting them is
# never a side effect of tidying up.
clean-results:
	$(PY) -c "import shutil; shutil.rmtree('results', ignore_errors=True); shutil.rmtree('figures', ignore_errors=True)"
