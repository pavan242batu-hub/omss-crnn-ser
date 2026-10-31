# Tested environment record

These files record the Colab environment captured after the repository
verification tests on 2026-10-31.

They describe the environment available at capture time. They are not
a record of the original historical manuscript-training environment,
and a fresh installation from these records has not been tested.

## Captured software versions

| Component | Version |
|---|---|
| Python | 3.13.16 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| librosa | 0.11.0 |
| SoundFile | 0.14.0 |
| Numba | 0.61.2 |
| h5py | 3.14.0 |

## Captured GPU information

- GPU reported by nvidia-smi: Tesla T4
- Reported total GPU memory: 15,360 MiB
- NVIDIA driver: 580.82.07

This is a capture-time hardware record, not independent proof of device
placement for every operation in the earlier tests.

## TensorFlow build information

TensorFlow reported:

- CUDA-enabled build: true
- Build CUDA version: 12.5.1
- Build cuDNN version: 9
- CPU compiler: clang
- ROCm build: false
- TensorRT build: false

Build information describes the TensorFlow binary. It should not be
interpreted as a separate inventory of all installed runtime libraries.

## Files

- `tested_environment.json`: structured package, platform, device,
  GPU-query, and TensorFlow-build information.
- `colab_package_snapshot.txt`: pip freeze output from the Colab
  environment, including packages unrelated to this repository.

The package snapshot is not a minimal requirements file or a validated
fresh-install specification. Do not treat it as proof that the full
Colab environment can be recreated unchanged on another system.

## Capture-time SHA-256 values

tested_environment.json:

769cd560fb5bb837230024b458cd98a06e67d320163d7b0832274e38c98634ea

colab_package_snapshot.txt:

fcb6bf0c17cb62274b585431cf0a2691c365266726ae7a1fa8654d6b207cea2a

These hashes identify the original captured files. Text normalization
during later edits may change their byte-level hashes.

## Verification boundary

The repository checks included complete 400-frame feature-cache
reconstruction, seven model construction/inference checks, five
validation-only fold checks, and one one-epoch training smoke test.

Full cross-validation, historical-result reproduction, and a clean
environment installation have not been verified.
