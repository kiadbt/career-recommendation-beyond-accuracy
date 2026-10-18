# Training and evaluation from source data

The commands below rebuild the catalogue, fixed encoder features, person partitions,
training examples, neural predictors, count baselines and saved rankings. They do not
load the authors' fitted predictors. “From scratch” refers to the career predictors;
the study uses the specified **pretrained** MiniLM encoder, which is not pretrained again.

## Environment

Use Python 3.12 and an isolated environment. For the original CUDA configuration:

```sh
python -m venv .venv
# Activate .venv using your operating system's normal activation command.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

The reference GPU is an NVIDIA RTX 2060 (6 GB). Commands use two CPU threads,
float32 computation, deterministic PyTorch operations, no TF32, and a maximum
4.5 GiB CUDA allocation. `--device cpu` is supported but can be substantially slower;
increase the explicit time cap deliberately if needed. Different PyTorch/CUDA/hardware
versions may change floating-point results. Compare predictions and metrics rather
than serialized checkpoint file hashes, which can differ despite identical tensors.

## Source assets

Review each upstream source's access conditions in `SOURCE_ASSETS.json`. The file
pins download URLs, revisions, sizes and SHA-256 checksums. It includes three JobHop
splits, the Decorte CSV files, KARRIEREWEGE's original test archive, Tabiya's taxonomy,
and the encoder configuration, tokenizer and weights. Approximately 593 MB is downloaded.

```sh
python code/fetch_assets.py --output-dir private/assets --accept-source-terms
```

Alternatively obtain the identical files from their owners and place them under the
names in the manifest. The downloader verifies existing files and does not silently
replace mismatched assets. It does not bypass authentication or access restrictions.
No career histories, user identifiers, predictions or trained weights should be committed.

## Rebuild inputs

```sh
python code/run_pipeline.py embeddings --assets private/assets --run-dir private/run
python code/run_pipeline.py prepare --assets private/assets --run-dir private/run
```

The encoder preserves the original text templates, 128-token limit, masked mean
pooling, normalization and batch composition. Training partitions are reconstructed
using the original SHA-256 person-selection rules; no private identifier list is needed.
Quarter sets retain all observed target occupations and all eligible fitting prefixes,
with each person's fitting weights summing to one. Chronological fitting ends in 2014;
the first eligible 2015 transition is used for tuning and the first eligible transition
from 2016 onward for evaluation. The two chronological evaluation samples are separate.

## Fit predictors

```sh
python code/run_pipeline.py panel --assets private/assets --run-dir private/run --max-seconds 1800
python code/run_pipeline.py chronological --assets private/assets --run-dir private/run --max-seconds 1800
```

The panel fits MLP, BiLSTM and self-attention for seeds 42 and 43. The chronological
stage fits BiLSTM for seeds 42–46. Popularity and quarter-transition statistics are
rebuilt from the respective fitting queries. Training uses batches of 128, gradient
clipping at 1, and tuning NDCG checkpoint selection. MLP/attention use AdamW
(learning rate 0.001, weight decay 0.01); BiLSTM uses RMSprop (learning rate 0.001).
Early stopping requires at least ten epochs and five nonimproving epochs, capped at
45. The original 30-epoch experiment and disclosed extension remain in the historical
records; the rebuild directly executes the final 45-epoch maximum specification.

Training refuses a nonempty output directory. A time-limit interruption retains
the best checkpoint, epoch records and last full state; it is not a successful run.
There is no automatic budget extension or automatic resume. `run.json` records
completion and resource use; individual training records report stopping adequacy.
For a short functional check, use `train_models.py --max-epochs 1` in a different
directory. Such a check does not reproduce the paper's full training runs.

## Score and evaluate

```sh
python code/run_pipeline.py score --assets private/assets --run-dir private/run
python code/report_panel.py --analysis primary --queries private/run/prepared/confirmation_queries.npz --predictions private/run/predictions/confirmation.npz --catalogue private/run/prepared/catalogue_texts.json --output private/run/primary_results.json
python code/report_panel.py --analysis fusion --queries private/run/prepared/fusion_queries.npz --predictions private/run/predictions/fusion.npz --catalogue private/run/prepared/catalogue_texts.json --output private/run/fusion_results.json
python code/report_chronological.py --prepared-dir private/run/prepared --predictions-dir private/run/predictions --trained-dir private/run/chronological --output private/run/chronological_results.json
```

Alternatively, `python code/run_pipeline.py report --assets private/assets --run-dir private/run`
produces the primary, fusion, Decorte, KARRIEREWEGE and chronological reports together.
Use it instead of the individual reporting commands above, because reports refuse overwrites.
`report_external.py` preserves the original source-specific bootstrap seeds and schemes,
including the KARRIEREWEGE unique nonmatching sequence sensitivity analysis.

Additional reported sensitivity analyses can be recalculated from the same predictions:

```sh
python code/report_sensitivity.py --prepared-dir private/run/prepared --predictions-dir private/run/predictions --output-dir private/run/sensitivity
python code/report_taxonomy.py --prepared-dir private/run/prepared --predictions-dir private/run/predictions --taxonomy private/assets/taxonomy.zip --output-dir private/run/taxonomy
python code/report_attention.py --source jobhop --queries private/run/prepared/confirmation_queries.npz --predictions private/run/predictions/confirmation.npz --output private/run/attention_jobhop.json
python code/report_attention.py --source decorte --queries private/run/prepared/decorte_queries.npz --predictions private/run/predictions/decorte.npz --output private/run/attention_decorte.json
python code/report_attention.py --source karrierewege --queries private/run/prepared/karrierewege/queries.npz --predictions private/run/predictions/karrierewege.npz --output private/run/attention_karrierewege.json
```

Scoring retains depth-50 rankings for the fusion-weight, input-depth and output-length
sensitivity analysis. The taxonomy analysis also reconstructs occupational hierarchy
checks from the source archive. These are exploratory analyses, as in the manuscript.

Scoring uses the full 3,007-occupation catalogue, deterministic ties, and the fixed
equal-weight reciprocal-rank fusion (constant 60, input depth 10). The confirmation
panel includes the history-neighbour control using fitting histories only.
The chronological report reproduces the original 5,000-draw history-only and joint
seed/history bootstrap calculations, including their seeds and seven-contrast family.
The primary report retains its original three-contrast family and the fusion report
its original four-contrast family. These families are not interchangeable.

`evaluate_fusion.py` can additionally evaluate saved rankings as documented in
`EVALUATION_SCOPE.md`; its seven-contrast history bootstrap is a general evaluation
interface and must not be substituted for a differently specified historical interval.

## Verify against the study reference

```sh
python code/verify_rebuild.py --run-dir private/run --output private/run/verification.json
```

`REBUILD_REFERENCE.json` contains hashes of the original array contents and trained
tensor contents, together with aggregate learning curves and selected epochs. The
checker verifies nine input archives and all eleven neural fits. Use `--inputs-only`
to verify the prepared inputs before training. Reference hashes disclose neither user-level
arrays nor model weights. A mismatch requires investigation; hardware-dependent
numerical differences should be reported rather than silently labelled an exact match.

## Scope of the rebuild

This workflow covers training all reported predictor architectures, rebuilding inputs
for all three datasets, the original six-model confirmation panel, fixed fusion,
external saved rankings and five-seed chronological analysis. It does not claim that
one command regenerates every exploratory ablation, diagnostic, figure or manuscript
layout. Archived aggregate exports remain available separately. See TRAINING_VALIDATION.json for the full runs and outputs verified locally.

The encoder and source data retain their upstream terms. Extracted STEP baseline
classes retain their MIT notice in `THIRD_PARTY_STEP_LICENSE.txt`; this does not license
the rest of this repository. The study-code licence remains subject to author/institution
confirmation. A local rerun by the authors is computational reproduction, not an
independent replication by another research group.
