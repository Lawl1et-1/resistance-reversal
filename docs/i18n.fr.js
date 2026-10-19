/* French translation. Keys match the data-i18n attributes in index.html.
   Values carry the same inner markup as the English, including the
   <span class="num" data-field="..."> placeholders, which are refilled from data.json
   after every language switch. Narrow no-break spaces ( ) before : % and other
   high punctuation, per French typographic convention. */

window.FR = {
  html: {
    "hero.h1": "Inverser la résistance in silico",
    "hero.lede": "Une cellule résistante au médicament et une cellule sensible sont deux états stables du même système. Ce travail apprend comment les perturbations déplacent une cellule de l’un à l’autre, cherche des interventions qui franchiraient la frontière, et calibre les cas où le modèle ne doit pas être cru.",
    "key.res": "<i class=\"key-res\"></i> bassin résistant",
    "key.sen": "<i class=\"key-sen\"></i> bassin sensible",
    "key.sep": "séparatrice",
    "key.note": "simulateur synthétique · in silico uniquement · sans relecture par les pairs",

    "p1.annot": "Planche I · le mécanisme",
    "p1.h2": "Inverser, c’est franchir un bassin",
    "p1.a": "Le simulateur est une bascule : deux régulateurs maîtres, chacun s’activant lui-même et réprimant l’autre. D’où deux états stables séparés par une frontière. Une perturbation ne pousse pas l’expression : elle déplace la frontière, et la cellule tombe dans le bassin où elle se retrouve alors.",
    "p1.b": "Ci-dessous ce système, intégré en direct à partir des équations du modèle. Déplacez la cellule. Inactivez un régulateur et observez la séparatrice bouger.",
    "p1.cap": "Axe horizontal : régulateur maître de la résistance. Vertical : régulateur maître de la sensibilité. Les cercles pleins sont des points fixes stables ; le cercle ouvert est le point instable sur la séparatrice. La trajectoire est intégrée par la méthode d’Euler à partir de la règle de mise à jour utilisée dans les expériences. <strong>Synthétique.</strong>",

    "p2.annot": "Planche II · premier résultat",
    "p2.h2": "La calibration tient, puis cède à la frontière, d’un seul coup",
    "p2.a": "Le modèle produit des intervalles de prédiction conforme, censés contenir la vérité <span class=\"num\" data-field=\"nominalPct\">90</span> % du temps. Sur des perturbations tenues à l’écart, c’est presque exactement le cas : <span class=\"num\" data-field=\"bestCoverage\">—</span>.",
    "p2.b": "Passez à un <strong>contexte cellulaire jamais vu</strong> et trois choses cèdent ensemble : la couverture tombe à <span class=\"num\" data-field=\"shiftedCoverage\">—</span>, l’estimation d’incertitude ne vaut pas mieux qu’un tirage à pile ou face pour signaler ses propres erreurs (AUROC <span class=\"num\" data-field=\"shiftedAuroc\">—</span>), et s’abstenir sur les prédictions les moins sûres cesse d’aider.",
    "p2.c": "C’est ce que prédit la théorie conforme lorsque son hypothèse d’échangeabilité est rompue : il s’agit donc d’une <em>confirmation</em>, non d’une découverte. Ce qui frappe, c’est que les trois propriétés de sûreté cèdent à la même frontière : le dispositif d’incertitude n’avertit en rien là précisément où il cesse de fonctionner.",

    "p3.annot": "Planche III · deuxième résultat",
    "p3.h2": "La mesure choisie décide du modèle gagnant",
    "p3.a": "Classez les modèles selon la métrique de précision conventionnelle, puis selon leur aptitude réelle à ordonner les interventions candidates : le gagnant change dans <span class=\"num\" data-field=\"rankChanges\">—</span> comparaisons sur <span class=\"num\" data-field=\"rankOf\">—</span>.",
    "p3.b": "La ligne pointillée ci-dessous est un modèle témoin ajusté sur des étiquettes de perturbation <strong>mélangées</strong>, qui par construction ne peut savoir quelle perturbation a été appliquée. Il obtient tout de même environ la moitié de la métrique conventionnelle. Une bonne part de ce que cette métrique récompense ne demande pas de connaître la perturbation.",

    "p4.annot": "Planche IV · troisième résultat, préliminaire",
    "p4.h2": "Le modèle de flot s’effondre là où son jumeau tient",
    "p4.a": "Sur des contextes cellulaires jamais vus, le modèle de flot obtient <span class=\"num\" data-field=\"flowContext\">—</span>, contre <span class=\"num\" data-field=\"ridgeContext\">—</span> pour la régression ridge. Un modèle par ailleurs identique, mais qui saute l’étape d’intégration itérative, obtient <span class=\"num\" data-field=\"mlpContext\">—</span>.",
    "p4.b": "Les deux ne diffèrent que par l’intégration ou non d’une équation différentielle sur vingt pas. Une vérification complémentaire, réduire le budget d’intégration à l’inférence à poids fixés, récupère la perte sur la partition décalée tout en laissant plat un témoin non décalé. Cela pointe vers une accumulation d’erreur au fil de l’intégration sous changement de distribution.",
    "p4.c": "<strong>Aucune cause n’est affirmée.</strong> L’explication évidente, un plongement de contexte non entraîné, prédit qu’un entraînement plus long ne change rien ; or un entraînement plus long a mesurablement nui. Les éléments à l’appui n’existent qu’à petite échelle sur une seule graine : c’est donc rapporté comme une observation, pas comme un résultat.",

    "p5.annot": "Planche V · verdicts",
    "p5.h2": "Ce que ce travail n’a pas pu conclure",
    "p5.a": "Deux des questions principales ne se sont pas tranchées. Elles reçoivent ici le même poids que les résultats, car la réponse honnête à une question non résolue n’est pas une version plus discrète d’une réponse.",

    "p6.annot": "Planche VI · corrections",
    "p6.h2": "Les affirmations qui n’ont pas survécu",
    "p6.a": "Treize défauts ont été trouvés et consignés pendant la construction. Quatre d’entre eux auraient produit un résultat convaincant mais faux. Voici les affirmations qui ont été écrites puis retirées. Elles restent affichées plutôt que supprimées, car les corrections sont la preuve que les chiffres survivants ont été vérifiés.",

    "p7.annot": "Planche VII · la recherche",
    "p7.h2": "Pourquoi la recherche d’interventions n’a pas pu être testée",
    "p7.a": "Le protocole devait comparer une recherche aveugle à l’incertitude à une recherche qui la pénalise, en attendant que la pénalité aide. Impossible à tester ici : sans aucune pénalité, la recherche atteint déjà <span class=\"num\" data-field=\"blindPct\">—</span> de la meilleure intervention possible. Il ne reste rien à récupérer pour une pénalité : elle ne peut que nuire, et à la plus forte valeur elle nuit.",
    "p7.cauh": "C’est un défaut de l’expérience, pas un résultat sur l’idée",
    "p7.caup": "Un protocole où la référence réussit déjà ne peut pas discriminer entre méthodes. Le tester correctement demande un problème de recherche plus difficile, un espace de candidats plus grand ou un modèle plus faible, et ce régime devrait être spécifié par écrit avant d’être exécuté, plutôt que construit jusqu’à ce que l’hypothèse apparaîsse.",

    "p8.annot": "Planche VIII · limites",
    "p8.h2": "Ce que ceci n’est pas",
    "p8.cauh": "Aucune validation expérimentale. Aucune affirmation biologique ou clinique.",
    "p8.caup": "Chaque chiffre de cette page provient d’un simulateur écrit pour ce projet. Sa structure encode une hypothèse sur le comportement des états de résistance ; ce n’en est pas une mesure. Rien ici n’a été testé sur des cellules, et l’objet d’étude est un <em>déplacement prédit vers un état transcriptionnel sensible au médicament</em> : pas un traitement, pas une guérison, et pas une démonstration qu’une résistance puisse être inversée dans quoi que ce soit de vivant.",
    "p8.more": "Autres limites à énoncer clairement : le score de sensibilité est un proxy appris dont les deux lectures indépendantes divergent fortement sur les états post-perturbation prédits. Seules cinq graines ont été exécutées, ce qui ne peut atteindre la significativité conventionnelle sous le test apparié employé ; les résultats sont donc donnés en intervalles et en tailles d’effet. Enfin, plusieurs règles d’analyse ont été révisées après l’arrivée des résultats, consignées avec leurs verdicts contrefactuels plutôt que présentées comme une histoire propre.",

    "f.repro": "Reproduire",
    "f.reprop": "Graines fixées partout. Chaque fichier de résultats consigne le commit git, la configuration et les versions des bibliothèques, et porte une étiquette de provenance <code>real</code> ou <code>synthetic</code>. Le profil complet demande environ neuf heures de CPU.",
    "f.docs": "Documents",
    "f.docsp": "<strong>LOG.md</strong> : les décisions et, longuement, les échecs.<br><strong>DECISIONS.md</strong> : chaque règle d’analyse modifiée après l’arrivée des résultats, avec ce que la règle initiale aurait conclu.<br><strong>MODEL_CARD.md</strong> : usage prévu, et ce à quoi ceci ne doit pas servir.",
    "f.src": "Source",
    "f.srcp": "Toutes les figures de cette page sont tirées de <code>results/full/</code> à la construction.",
    "f.repo": "Dépôt →",
  },

  /* Strings the charts, tables and ledger generate at runtime. */
  title: "Inverser la résistance in silico — recherche contrefactuelle calibrée sur les états cellulaires",

  ui: {
    splits: {
      unseen_single: "Gène seul, jamais vu",
      unseen_combo_seen_genes: "Paire inédite, gènes vus",
      unseen_combo_unseen_genes: "Paire inédite, gènes jamais vus",
      unseen_context: "Contexte cellulaire, jamais vu",
    },
    models: {
      additive: "Additif",
      ridge_embedding: "Ridge (plongements de gènes)",
      low_rank_linear: "Linéaire de rang faible",
      pertflow: "PertFlow (flot)",
      pertflow_mlp: "PertFlow (sans intégration)",
      shuffled_labels: "Étiquettes mélangées",
      control_mean: "Témoin sans changement",
      oracle_latent: "Oracle latent",
    },
    shuffledControl: "témoin mélangé",
    assumptionViolated: "hypothèse violée",
    nominal: "nominal",
    na: "s.o.",
    accCaption: "Delta de Pearson par partition · 5 graines · intervalles hiérarchiques à 95 %",
    searchCaption: "Recherche d’interventions · glouton · vérifiée contre l’oracle du simulateur",
    colModel: "Modèle",
    colLambda: "Pénalité d’incertitude λ",
    colVerified: "Inversion vérifiée",
    colInterval: "Intervalle à 95 %",
    colRegret: "Regret vs meilleur possible",
    states: { supported: "confirmée", inconclusive: "non concluante", untestable: "non testable" },
    kdResistance: "Pilote de résistance",
    kdSensitivity: "Pilote de sensibilité",
    kdLabel: (id) => `INACT. ${id}`,
    provenance: (p, seeds, src, commit) =>
      `données ${p === "synthetic" ? "synthétiques" : p} · ${seeds} graines · générées depuis ${src}`
      + (commit ? ` · commit ${commit}` : ""),
    ledger: {
      h1a: {
        claim: "Les modèles profonds ne battent pas de manière fiable les références linéaires.",
        why: (v) => `${v.wins} victoires nettes et ${v.losses} défaites nettes sur ${v.n} comparaisons appariées ; avantage maximal +0,097, déficit maximal −0,605. Le verdict bascule selon la définition de « battre de manière fiable », et cette définition a été choisie après avoir vu les comptes : il est donc rapporté comme non résolu.`,
      },
      h1b: {
        claim: "Une évaluation orientée inversion réordonne les modèles.",
        why: (v) => `Le gagnant change dans ${v.changes} comparaisons sur ${v.of} entre la métrique conventionnelle et une métrique qui mesure l’ordonnancement des interventions.`,
      },
      h2: {
        claim: "La calibration conforme atteint sa cible sur des perturbations inédites, et l’abstention réduit l’erreur.",
        why: (v) => `Vraie sur les ${v.valid} partitions où l’hypothèse d’échangeabilité s’applique. Elle échoue, comme la théorie le prévoit, sur celle où elle ne s’applique pas.`,
      },
      h3: {
        claim: "Pénaliser l’incertitude améliore la recherche d’interventions.",
        why: (v, pctFn) => `λ = 0 atteint déjà l’oracle à 1 % près, donc aucune pénalité ne peut améliorer la recommandation, quelle que soit la manière de mesurer l’optimisme. La recherche aveugle à l’incertitude atteint déjà ${pctFn(1 - v.blind_regret)} de la meilleure intervention possible. C’est un défaut du protocole, pas une preuve contre l’idée.`,
      },
      h45: {
        claim: "La sélection active bat le hasard ; les cibles les mieux classées sont enrichies en biologie connue.",
        why: () => "Non exécuté. H5 nécessite des données réelles, que ce projet n’a jamais acquises.",
      },
    },
  },
};
