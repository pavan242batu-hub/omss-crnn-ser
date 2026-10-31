# Dataset inputs and verification status

## Scope

This document records structural checks of the manifest and
fold-assignment CSV used in the portable workflow.

These checks establish internal consistency of the supplied records.
They do not independently establish audio provenance, the historical
dataset-selection procedure, or OMSS enhancement provenance.

Raw audio and feature caches are not included in this repository.
The input CSVs inspected during these checks were held separately
in Google Drive.

## Manifest

Filename: `omss_grouped_manifest.csv`

Required columns in the inspected file:

- `FilePath`
- `FileName`
- `OriginalID`
- `Emotion`
- `NoiseType`
- `SNR`
- `Split`

Verified properties:

- 1,104 rows representing 69 unique original recordings.
- No missing `OriginalID` values.
- No duplicate `FileName` values.
- Exactly 16 rows per original recording.
- Each original has exactly one row for every combination of:
  - Noise: `aircraft`, `car`, `traffic`, `train`.
  - SNR: `0db`, `5db`, `10db`, `15db`.
- Each original has one consistent emotion value.

These checks concern manifest records; they do not independently
verify the acoustic noise type or SNR of the corresponding audio.

## Fold assignments

Filename: `cv_fold_assignment.csv`

Columns:

- `OriginalID`
- `y`
- `fold`

Verified properties:

- 69 rows and 69 unique original IDs.
- No missing or duplicate original IDs.
- The original-ID set exactly matches the manifest.

Audited label mapping:

| Numeric label | Emotion | Original recordings |
| --- | --- | --- |
| 0 | angry | 26 |
| 1 | frustrated | 24 |
| 2 | neutral | 6 |
| 3 | sad | 13 |

Outer-fold assignment counts, measured in original recordings:

| Fold | Angry | Frustrated | Neutral | Sad |
| --- | --- | --- | --- | --- |
| 0 | 6 | 6 | 0 | 2 |
| 1 | 7 | 5 | 0 | 2 |
| 2 | 5 | 4 | 2 | 2 |
| 3 | 3 | 5 | 2 | 4 |
| 4 | 5 | 4 | 2 | 3 |

## Split fields are not interchangeable

The manifest's `Split` column contains:

- `train`: 752 rows.
- `validation`: 128 rows.
- `test`: 224 rows.

These counts describe the supplied manifest field. They do not
describe the five outer-fold partitions.

The portable training workflow uses the assignment CSV to select
the outer test fold and reconstructs training and validation
partitions as documented in the reproducibility guide.

The historical purpose and generation procedure of the manifest's
`Split` field remain to be documented.

## Evaluation limitations

Neutral originals are absent from outer test folds 0 and 1.
Additional validation-partition class omissions and the implications
of the preserved four-label macro-recall metric are documented in
[Reproducibility and verification status](REPRODUCIBILITY.md).

Grouping by `OriginalID` does not establish speaker-independent
evaluation.

## Reconstruction information still required

This is an input-structure record, not a complete reconstruction guide.

Remaining documentation includes:

- The exact source-recording selection list and selection procedure.
- Source-data access requirements and applicable permissions.
- The procedure and source assets used to generate noisy variants.
- The historical OMSS implementation and settings used for enhancement.
- Instructions for recreating the manifest and fold assignments.
- An authorized distribution or reconstruction route for required inputs.

Do not infer missing procedures from filenames or category counts.
