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
| `chronological_results/` | Five-seed chronological results, selection records, protocol and verification records |
| `supplementary_tables/` | Ten detailed supplementary tables, with a file index and interpretation notes |
| `code/` | Source-data preparation, predictor training, scoring and statistical reports |
| `TRAINING.md` | Commands to rebuild inputs, train all predictor architectures and rerun analyses |
| `SOURCE_ASSETS.json` | Pinned source downloads and checksums |
| `REBUILD_REFERENCE.json` | Array/tensor reference hashes and aggregate learning curves |
| `protocols/` | Recorded analysis protocols |
| `PROVENANCE.json` | Source URLs, revisions and required-asset hashes |
| `REPRODUCING.txt` | Local input schemas, commands and tested scope |
| `VALIDATION.json` | Historical local validation record; its publication flag refers to the version before public release |

`cohort_shift` and `Delta_cohort` in archived machine-readable records correspond to **change in model gap** in the manuscript. Existing field names are retained for compatibility. All supplementary table values preserve the displayed precision of the Word tables; JSON result files retain greater precision.

## Saved-ranking evaluation

`code/evaluate_fusion.py` calculates complete-target NDCG, repeated/broader recall and seven paired contrasts from local saved rankings. See [EVALUATION_SCOPE.md](EVALUATION_SCOPE.md) for commands, validation and the distinction between history-only and joint seed/history uncertainty. Run the saved-ranking checks with `python -m unittest discover -s tests -p test_evaluate_fusion.py -v` after installing NumPy. No GPU is required.

## Data access and reproducibility limits

Obtain source data from the owners under their applicable conditions:

- [JobHop v2](https://huggingface.co/datasets/aida-ugent/JobHop)
- [Decorte anonymous working histories](https://huggingface.co/datasets/TechWolf/anonymous-working-histories)
- [KARRIEREWEGE](https://huggingface.co/datasets/ElenaSenger/Karrierewege)

This repository contains research code and aggregate records, not raw career histories,
personal identifiers, per-history predictions, catalogue text, embeddings or trained
weights. The training workflow in [TRAINING.md](TRAINING.md) reconstructs catalogue
features and person partitions from pinned source assets, trains MLP, BiLSTM and
self-attention predictors, and produces rankings and statistical reports. No private
identifier list or author-trained checkpoint is needed for that workflow. The MiniLM
encoder remains a fixed pretrained input; it is not pretrained again.

The workflow covers the original panel, external scoring, five-seed chronological
refitting and principal sensitivity analyses. It does not automatically rebuild every
exploratory diagnostic or the manuscript's figures and layout. Source access conditions
and numerical differences across software/hardware environments remain relevant.
See [TRAINING_VALIDATION.json](TRAINING_VALIDATION.json) for the tested scope: nine input archives, eleven exact fitted-tensor matches, prediction replay and scientific-report comparisons.

Use `requirements-training.txt` for the new training workflow, installing the matching
PyTorch build separately. Older preprocessing/analysis requirement files describe
earlier tool environments; do not combine all requirement files. Aggregate checks
still need only the Python standard library and no GPU.

Original and post hoc analyses are distinguished in the protocols. KARRIEREWEGE uses ordered singleton experiences; the other sources use quarter-level sets. Cross-source person independence is not verified. Historical validation records describe checks with private local assets; publication of a record does not distribute those assets or make the checks independently reproducible.

## Citation and permissions

Use `CITATION.cff` to cite this research software/records release. No manuscript DOI is available yet. No open-source licence is asserted pending author/institutional confirmation. Public visibility does not extend the licences of upstream datasets, models or software. See `RIGHTS.txt`.
