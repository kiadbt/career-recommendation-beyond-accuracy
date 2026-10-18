# Career Recommendation Beyond Overall Accuracy

Reproducibility records for **Career Recommendation Beyond Overall Accuracy: Model Comparisons and Broader Occupational Retrieval**.

Authors: Kamaluddeen Ibrahim Yarima, Abdullahi Abubakar Imam, Rosyzie Anna Apong, Nazir Muhammad Ilyasu and Usman Ali.

This study compares occupation-ranking models on JobHop v2, Decorte and KARRIEREWEGE. It examines differences between evaluation groups, retrieval of broader observed occupational moves, external transfer and chronological validation. The manuscript is unpublished; this repository does not imply journal acceptance.

## Quick verification

Python 3.12 or later, standard library only:

```sh
python export_tables.py
python verify_release.py
```

The first command verifies package hashes and aggregate arithmetic and exports ten tables from the original and external analyses. The second checks the five-seed chronological contrast means and standard deviations and exports the chronological contrasts, including stored uncertainty intervals. Neither command reruns training or reconstructs bootstrap intervals from individual histories.

## Contents

| Location | Contents |
| --- | --- |
| `aggregate_results/` | Original model comparisons, fusion, external transfer, self-attention and sensitivity estimates |
| `extended_results/` | Cross-source diagnostics, composition and occupational-grouping sensitivity |
| `chronological_results/` | Five-seed chronological results, selection records, protocol and verification receipts |
| `supplementary_tables/` | Ten detailed tables from the condensed supplement, with a file index and interpretation notes |
| `code/` | Portable preprocessing, recency scoring, self-attention inference and paired evaluation code |
| `protocols/` | Recorded analysis protocols |
| `PROVENANCE.json` | Source URLs, revisions and required-asset hashes |
| `REPRODUCING.txt` | Local input schemas, commands and tested scope |
| `VALIDATION.json` | Historical local validation receipt; its publication flag describes the pre-release candidate |

`cohort_shift` and `Delta_cohort` in archived machine-readable records correspond to **change in model gap** in the manuscript. Existing field names are retained for compatibility. All supplementary table values preserve the displayed precision of the Word tables; JSON result files retain greater precision.

## Saved-ranking evaluation

`code/evaluate_fusion.py` calculates complete-target NDCG, repeated/broader recall and seven paired contrasts from local saved rankings. See [EVALUATION_SCOPE.md](EVALUATION_SCOPE.md) for commands, validation and the distinction between history-only and joint seed/history uncertainty. Run the synthetic checks with `python -m unittest discover -s tests -v` after installing NumPy. No GPU is required.

## Data access and reproducibility limits

Obtain source data from the owners under their applicable conditions:

- [JobHop v2](https://huggingface.co/datasets/aida-ugent/JobHop)
- [Decorte anonymous working histories](https://huggingface.co/datasets/TechWolf/anonymous-working-histories)
- [KARRIEREWEGE](https://huggingface.co/datasets/ElenaSenger/Karrierewege)

This repository contains aggregate records and code, not raw career histories, personal identifiers, per-history predictions, catalogue text, embeddings or trained weights. Exact scoring requires separately available fitted assets identified in `PROVENANCE.json`. The portable inference command covers self-attention; it is not a complete training pipeline for every reported model. Chronological results are released as aggregates and protocol records, not a standalone chronological training implementation. End-to-end raw-data-to-model reproduction has not been verified in a clean environment.

Preprocessing and analysis requirements are separate environment snapshots. Do not install both requirement files into one environment without reconciling their NumPy versions. Aggregate checks need neither file. Additional GPU training is unnecessary for those checks.

Original and post hoc analyses are distinguished in the protocols. KARRIEREWEGE uses ordered singleton experiences; the other sources use quarter-level sets. Cross-source person independence is not verified. Historical validation receipts describe checks with private local assets; publication of a receipt does not distribute those assets or make the checks independently reproducible.

## Citation and permissions

Use `CITATION.cff` to cite this research software/records release. No manuscript DOI is available yet. No open-source licence is asserted pending author/institutional confirmation. Public visibility does not extend the licences of upstream datasets, models or software. See `RIGHTS.txt`.
