# OMSS–CRNN Speech Emotion Recognition

## Reproducibility and verification

See [Reproducibility and verification status](docs/REPRODUCIBILITY.md)
for commands, required inputs, tested settings, and known limitations.

Completed checks include:

- Exact reconstruction of all 1,104 noisy and 1,104 OMSS feature arrays
  in the existing 400-frame caches.
- Construction and inference checks for all seven model configurations.
- Cache-alignment and OriginalID-separation checks for all five outer folds.
- One successful one-epoch end-to-end training smoke test for
  `fusion_guided`, fold 0, seed 42.
- SHA-256 verification that the four scripts downloaded from GitHub
  match the tested local versions.

These checks do not establish full cross-validation performance or
reproduction of manuscript results.

The supplied assignments omit neutral examples from some validation
and test partitions. The historical four-label macro-recall calculation
is preserved; see the reproducibility document for its implications.

The feature extractor requires supplied noisy and already-enhanced OMSS
audio. It does not perform OMSS enhancement. 

Research code for the manuscript:

“Modulation-Domain Enhanced Speech Emotion Recognition
for IoT Edge Devices: An OMSS and CRNN-Attention Approach.”

## Current status

This private repository is being prepared and verified.

It will organize the recognition, training, and evaluation code
associated with the manuscript. It is not yet a verified,
complete reproduction package.

## Pipeline

Noisy speech and corresponding OMSS-enhanced speech
→ MFCC extraction
→ single-stream or paired-stream CRNN
→ temporal aggregation
→ emotion prediction
→ grouped evaluation.

## Implementation boundary

The recognition and evaluation components are implemented
in Python.

The currently inspected OMSS source is MATLAB-based.
A validated Python port of OMSS is not currently claimed.

The exact historical OMSS source version used to generate
the complete enhanced dataset has not been fully verified.

## Dataset and evaluation scope

- 69 original utterances from an IEMOCAP Session-1 subset.
- Four emotion classes: angry, frustrated, neutral, and sad.
- Four noise conditions and four SNR levels.
- 16 acoustic variants per original utterance.
- 1,104 recordings per input condition.
- Original-utterance-grouped evaluation.

This evaluation does not establish speaker-independent
generalization.

## Experimental configurations

The manuscript contains separate experiments with different
input lengths, model configurations, and metric summaries.

These experiments will be documented separately rather than
treated as a single interchangeable configuration.

## Data availability

Raw audio is not included in this repository.

Dataset sources, selected recording identifiers, manifests,
and reconstruction instructions will be documented separately.
Redistribution will depend on the applicable source permissions.

## Reproducibility

Dependency versions, verified execution commands, and mappings
between scripts and manuscript results will be added after
checking the selected implementation files.

## Access

This repository is private and intended for authorized
research collaborators during preparation.
