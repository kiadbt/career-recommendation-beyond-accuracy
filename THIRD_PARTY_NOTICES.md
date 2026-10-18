# Third-party code

`code/baseline_models.py` contains the `MLPTransformation` and `CareerBiLSTM`
classes extracted without changing their definitions from
[STEP baselines.py](https://github.com/aida-ugent/Step/blob/158e2e81def9aa5874e3fc76a9e660ed3d2a90e9/step/predictor/baselines.py).
The surrounding application imports were replaced with the minimal PyTorch imports.
The original MIT copyright and permission notice is reproduced in
`THIRD_PARTY_STEP_LICENSE.txt`. That licence applies to those upstream classes;
it does not grant a licence to the rest of the study package.

Dataset and pretrained-model permissions remain with their owners. See
`SOURCE_ASSETS.json`, `PROVENANCE.json`, and `RIGHTS.txt`.
