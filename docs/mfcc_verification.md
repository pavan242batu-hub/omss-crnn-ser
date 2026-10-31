# MFCC Cache Verification

## Status
This document records the earlier sample-only verification. A later full-cache comparison using the portable extractor exactly reproduced all 1,104 noisy and all 1,104 OMSS arrays at 400 frames, with maximum absolute difference zero. See [Reproducibility and verification status](REPRODUCIBILITY.md) for the later verification scope and limitations.

The sample procedure, environment, and results below are retained as the record of the earlier check.

A six-recording sample check passed for the existing
400-frame noisy and OMSS feature caches.

All six rebuilt arrays exactly matched their corresponding
cached arrays: maximum and mean absolute differences were zero.

This is a sample verification, not full-dataset verification.

## Recovered source

Notebook: SERRevision_IoT_Eval_v1.ipynb
Code cell at extraction: 98

Source-cell SHA-256:
662cc370b35a9ec8398c181bbcbb9fae0b42e847d4bdfe138d938d274f722b3c

Preserved source:
docs/legacy/mfcc_long_notebook_cell.py

## Feature procedure

- Audio loaded using librosa at 16,000 Hz, mono.
- 40 MFCC coefficients.
- FFT length: 512 samples.
- Window length: 512 samples.
- Hop length: 160 samples.
- Mel bands: 64.
- Centered analysis: enabled.
- Per-recording, per-coefficient mean/std normalization.
- Normalization is performed over the complete extracted
  feature sequence before padding or truncation.
- Normalization denominator includes an epsilon of 1e-8.
- Short sequences are zero-padded at the end.
- Long sequences retain their first 400 frames.
- Output dtype: float32.
- Output shape per recording: (400, 40, 1).
- Other feature and audio-loading options follow the recovered
  function's library defaults in the environment recorded below.

## Verification environment

- Python: 3.13.16
- NumPy: 2.1.3
- pandas: 2.2.3
- librosa: 0.11.0
- SoundFile: 0.14.0
- SciPy: 1.16.3

These are the versions used for this verification run.
They are not independently established as the original
cache-generation environment.

## Tested recordings

Zero-based manifest row indices: 0, 552, and 1103.

Each recording was tested in both noisy and OMSS conditions:

- Ses01M_script01_1_F008_aircraft_0db.wav
- Ses01M_script01_1_M010_traffic_0db.wav
- Ses01M_script01_1_M042_train_5db.wav

## Comparison

The rebuilt features were compared with the corresponding
manifest-row entries of:

- mfcc_noisy_T400.npz
- mfcc_omss_T400.npz

Both caches had the expected shape:
(1104, 400, 40, 1).

Comparison tolerances:
rtol = 1e-4
atol = 1e-4

Observed differences:
maximum absolute difference = 0
mean absolute difference = 0
for every sampled comparison.

## Limitations

- The remaining cache entries were not checked in this earlier sample verification; the later full-cache comparison is documented in the reproducibility guide. 
- This test does not verify the main 128-frame feature caches.
- This test does not establish historical OMSS implementation
  provenance.
- This test does not reproduce classifier training or results.
- No training was performed and existing caches were not changed.
