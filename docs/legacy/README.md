# Inspected Historical Source

This directory preserves inspected research scripts without
modifying their implementation.

## Files

### long_window_cv.py

Defines single-stream and shared-encoder fusion CRNN models
using bidirectional GRUs, average pooling, temporal attention,
and guided attention.

It provides data-loading and fold-training functions but does
not contain a standalone experiment-launching entry point.

SHA-256:
aa56d8770c01936a6bcb61a3ee2977fbfd17c446a5aa2162f6f5c5c8756f79e7

### channel_robustness_cv.py

Contains the same model and fold-training functions, with
additional saving of trained model weights.

Despite its filename, this source does not itself implement
telephone-band processing or the full channel-evaluation workflow.

SHA-256:
d22efd96a0084b363fa197ef1e41a1f35e471e3d69cbded14ec64c7515219094

## Existing dependencies

Both scripts contain fixed Colab project paths and load
precomputed noisy and OMSS MFCC caches.

The cache-generation procedure and experiment-launch commands
must be recovered and documented separately.

## Provenance boundary

Preservation of these files does not establish that either
script generated every historical result in the manuscript.

Any portable implementation will be maintained separately
and checked against these preserved sources and saved results.
