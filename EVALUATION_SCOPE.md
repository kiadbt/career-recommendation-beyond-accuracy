# Evaluation coverage

The package has three distinct levels of reproducibility. They should not be treated as interchangeable.

| Command | Required inputs | What it checks or calculates |
| --- | --- | --- |
| `python export_tables.py` | Included aggregate JSON files | File integrity, selected arithmetic identities and table exports using stored intervals |
| `python verify_release.py` | Included chronological JSON files | Five-seed means and standard deviations; exports stored contrast summaries |
| `python code/evaluate_fusion.py ...` | Locally available queries, saved rankings and occupation-group mapping | Full-target NDCG, repeated/broader target recall, seven paired contrasts and optional history-bootstrap intervals |

The portable self-attention inference and three-contrast evaluation commands remain documented in `REPRODUCING.txt`. Neither inference path supplies missing trained weights or reconstructs training.

## Evaluating neural, recency and fusion rankings

Use the query schema in `REPRODUCING.txt`. All target and history occupations must map to the supplied catalogue. `groups.json` is a JSON list containing one occupational-group label per catalogue row, in exactly the same index order as the saved rankings. For the primary definition use the verified two-digit ISCO mapping. The program rejects unmapped indices instead of silently dropping targets. Do not infer group membership from arbitrary numeric catalogue indices.

The prediction NPZ must include `pid` and one `(N,k)` integer array per ranking. Every ranking must contain distinct catalogue indices. Query and prediction identifiers must match in order. The example below uses two saved seeds; chronological evaluation uses keys for seeds 42 through 46.

```sh
python code/evaluate_fusion.py --queries queries.npz --predictions predictions.npz --groups groups.json --neural-keys top_BiLSTM_42 top_BiLSTM_43 --fusion-keys top_Fusion_42 top_Fusion_43 --draws 5000 --seed 20261021 --out fusion_evaluation.json
```

The recency key defaults to `top_PersonalRecency`. Use `--draws 0` to calculate point estimates only. Paths refer to locally obtained assets; the repository does not supply them. The command evaluates supplied fusion rankings; it does not construct or tune a fusion rule.

NDCG retains the complete observed target set. Unseen-positive histories have at least one target occupation absent from their entire input history. Repeated-target recall uses repeated occupations as its denominator. Broader-target recall uses target occupations whose group is absent from the entire history; histories with no eligible target are excluded from that recall. This is observed-target recovery, not list diversity or career suitability.

Seed-specific metrics are averaged within history. Bootstrap draws resample histories jointly across all models and recompute eligible denominators. A maximum absolute centred deviation supplies a simultaneous radius over the seven contrasts for that sample. The command does not resample training seeds; its intervals must not be labelled as the manuscript's joint seed/history intervals. It does not correct for architecture selection or the adaptive research history. Different seeds, bootstrap algorithms or contrast families can produce different interval radii.

## Validation

```sh
python -m unittest discover -s tests -v
```

Synthetic tests cover full targets, history-wide occupational grouping, paired contrasts, duplicate candidates, identifier misalignment and empty eligible subsets. They contain no source records. Local comparison additionally reproduced all seven published chronological point estimates on the 9,551-history confirmation sample and 9,671-history fusion sample, using all five saved seeds, to absolute tolerance 1e-12. `EVALUATION_VALIDATION.json` records that check; its private inputs are not distributed. No bootstrap equivalence or independent scientific replication is claimed by that receipt.

## Remaining limits

Complete original/chronological training, historical catalogue reconstruction and unrestricted fitted-asset distribution remain outside this package. Third-party access conditions and a pending code-licence decision remain applicable. The public code and records improve auditability but do not constitute a fully reproducible raw-data-to-model pipeline.
