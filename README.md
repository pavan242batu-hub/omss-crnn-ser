# OMSS–CRNN Speech Emotion Recognition

Research code associated with the manuscript:

“Modulation-Domain Enhanced Speech Emotion Recognition
for IoT Edge Devices: An OMSS and CRNN-Attention Approach.”

## Current status

This private repository contains a portable MFCC feature extractor,
preserved convolutional/BiGRU model definitions, historical fold-training
helpers, and a validation-first command-line training entry point.

Selected components have been tested as described below. The repository
is not yet a verified, complete reproduction of the manuscript results.

## Reproducibility and verification

See [Reproducibility and verification status](docs/REPRODUCIBILITY.md)
for commands, required inputs, tested settings, and known limitations.

Completed checks include:

- Exact reconstruction of all 1,104 noisy and 1,104 OMSS feature arrays
  in the existing 400-frame caches using the current audio and manifest.
- Construction and inference checks for all seven portable-interface
  model configurations.
- Cache-alignment and OriginalID-separation checks for all five supplied
  outer folds.
- One successful one-epoch end-to-end training smoke test for
  `fusion_guided`, fold 0, seed 42.
- SHA-256 verification that the four scripts downloaded from GitHub
  match the tested local versions.

These checks do not establish full cross-validation performance,
historical experiment provenance, or reproduction of manuscript results.

The supplied assignments omit neutral examples from some validation
and test partitions. The historical four-label macro-recall calculation
is preserved; see the reproducibility document for its implications.

## Repository entry points

| File | Purpose |
|---|---|
| `scripts/extract_features.py` | Extract MFCC features from supplied noisy or already-enhanced OMSS audio |
| `scripts/models.py` | Define the preserved convolutional/BiGRU models |
| `scripts/training_core.py` | Preserve the historical fold-training procedure |
| `scripts/train_fold.py` | Validate inputs and optionally train one configuration on one outer fold |

The training entry point defaults to validation only.
Training requires an explicit `--train` flag.

## Pipeline

Supplied noisy speech and corresponding OMSS-enhanced speech
→ MFCC extraction
→ single-stream or paired-stream convolutional/BiGRU model
→ temporal pooling or attention
→ four-class emotion prediction
→ original-utterance-grouped evaluation.

For paired-stream models, the input channel order is noisy first,
OMSS second.

## Implementation boundary

The portable recognition and fold-training components are implemented
in Python.

The feature extractor requires supplied noisy and already-enhanced
OMSS audio. It does not perform OMSS enhancement.

The currently inspected OMSS source is MATLAB-based.
A validated Python port of OMSS is not currently claimed.

The exact historical OMSS source version used to generate the complete
enhanced dataset has not been fully verified. Exact feature-cache
reconstruction does not establish that enhancement provenance.

## Dataset and evaluation scope

The project dataset is described as:

- 69 original utterances from an IEMOCAP Session-1 subset.
- Four emotion classes: angry, frustrated, neutral, and sad.
- Four noise conditions and four SNR levels.
- 16 acoustic variants per original utterance.
- 1,104 recordings per input condition.

The audited manifest and assignments contain 1,104 rows representing
69 unique OriginalIDs, with the following label mapping:

| Label | Emotion | Unique original utterances |
|---|---|---:|
| 0 | angry | 26 |
| 1 | frustrated | 24 |
| 2 | neutral | 6 |
| 3 | sad | 13 |

No OriginalID overlaps were found between training, validation, and
test within any of the five supplied outer folds.

This check establishes separation by the supplied OriginalID.
It does not establish speaker-independent generalization.

## Experimental configurations

The manuscript describes separate experiments with different input
lengths, model configurations, and metric summaries.

The completed feature-reconstruction and model checks documented here
use 400-frame inputs. They should not be generalized to other input
lengths or treated as verification of every manuscript experiment.

The seven portable-interface configuration names are documented in
the reproducibility guide. Their availability does not independently
establish which historical runs produced specific manuscript results.

## Tested environment

The model and training smoke tests ran with:

- Python 3.13.16
- TensorFlow 2.21.0
- Keras 3.13.2

Additional package versions, capture-time hardware information, and a full Colab package snapshot are available in [the environment record](docs/environment/README.md). These records describe the environment at capture time; they do not establish the hardware used for every earlier verification check. A fresh installation from these records has not been tested, and the package snapshot is not a validated dependency lockfile.

The successful one-epoch training run emitted a TensorFlow
layout-optimizer diagnostic. Training completed and saved-output checks
passed; the diagnostic and its verification limits are recorded in
the reproducibility guide.

## Data availability

Raw audio and feature caches are not included in this repository.

Dataset-source details, selected recording identifiers, and reconstruction
instructions remain to be completed. Users must supply the required
audio, manifest, and fold assignments to run the documented commands.

Redistribution will depend on the applicable source permissions.

## Remaining work

- Validate a fresh installation using the captured environment records and document any dependency or hardware constraints. 
- Document dataset selection and reconstruction details.
- Establish mappings between historical runs and manuscript results.
- Verify training beyond the single completed one-epoch configuration.
- Evaluate multi-epoch behavior and full cross-validation runs if needed.
- Verify model serialization if a saved-model workflow is added.

The preserved training core currently saves test predictions, not
a trained-model artifact or a full epoch-by-epoch training history.

## Access

This repository is private and intended for authorized research
collaborators during preparation.
