# Experiment Map

This document maps manuscript experiments to their implementation
files, configurations, and saved results.

Items marked "To verify" must be checked against the actual code
before this repository is described as a complete reproduction package.

## 1. Main integration comparison

- Configurations: OMSS only, noisy only, fusion with average
  pooling, and guided-attention fusion.
- Training seeds: 42, 52, and 62.
- Evaluation: five original-utterance-grouped outer folds.
- Primary summary: pooled out-of-fold UAR calculated separately
  for each seed, followed by the mean and sample standard
  deviation across seeds.
- Training script: To verify.
- Input length and exact architecture: To verify.
- Configuration file: To verify.

## 2. Corrected main bootstrap analysis

- Inputs: saved main-comparison out-of-fold predictions.
- Statistic: difference in mean per-seed pooled UAR.
- Resampling unit: original utterance, retaining all 16 variants.
- The same resample is applied across configurations and seeds.
- Valid bootstrap resamples: 2,000.
- Bootstrap random seed: 2026.
- Resamples omitting an emotion class are rejected and redrawn.
- Intervals: 95% percentile confidence intervals.
- No model retraining is performed during resampling.
- Analysis script: To add.

## 3. Separate 400-frame recognition experiment

- Input length: 400 MFCC frames.
- Training seed: 42.
- Evaluation: five original-utterance-grouped folds.
- Summary: unweighted mean of fold-level recognition metrics.
- Training and evaluation scripts: To verify.

## 4. Telephone-band experiment

- Input length: 400 MFCC frames.
- Training seed: 42.
- Additional channel processing is applied before enhancement.
- Evaluation is performed without channel-specific retraining.
- Summary: unweighted mean of fold-level recognition metrics.
- Processing and evaluation scripts: To verify.

## 5. Frozen WavLM baseline

- Frozen WavLM-Base-Plus representation and logistic regression.
- Summary: pooled out-of-fold UAR and macro-F1.
- The baseline is not a fully fine-tuned WavLM system.
- Exact checkpoint, pooling, scaling, classifier settings,
  and fold reuse: To verify against the execution code.
- Implementation script: To verify.

## 6. Classifier profiling

- Separate audited 128-frame BiLSTM checkpoints.
- Host-based classifier timing and model-export checks.
- Excludes OMSS and MFCC extraction from classifier timing.
- Does not establish end-to-end embedded-device performance.
- Profiling script and environment: To verify.

## 7. OMSS enhancement

- Currently inspected reference source: MATLAB implementation.
- Historical source-version linkage to the complete enhanced
  evaluation dataset remains unresolved.
- A validated Python OMSS port is not currently claimed.
- Source files and implementation notes: To add after review.

## Interpretation boundary

These experiments must not be treated as one interchangeable
model configuration or evaluation protocol.

Only verified implementation files and settings should be
used to replace the "To verify" entries.
