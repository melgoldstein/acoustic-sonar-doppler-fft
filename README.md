# Acoustic sonar Doppler features

[Portfolio: melgoldstein.com](https://melgoldstein.com/)

[![Verification](https://github.com/melgoldstein/acoustic-sonar-doppler-fft/actions/workflows/ci.yml/badge.svg)](https://github.com/melgoldstein/acoustic-sonar-doppler-fft/actions/workflows/ci.yml)

Extract spectral centroid, 85-percent energy roll-off and Hilbert-envelope blade
modulation frequency. Compare scaled logistic regression and RBF SVC on independently
seeded amplitude-modulated propeller signals with noise and frequency shifts.

```sh
pip install -e '.[test]'
sonar-classify
pytest -q
```

| CLI field | Meaning |
|---|---|
| balanced_accuracy | Mean recall on independently generated holdout trials |
| confusion_matrix | True rows, predicted columns: light, medium, heavy |
| train_seed / holdout_seed | Reproduction seeds, 31 / 87 |

| Model | Holdout balanced accuracy | Correct / trials |
|---|---:|---:|
| Logistic regression | 1.000 | 90 / 90 |
| RBF SVC | 1.000 | 90 / 90 |

The full confusion matrices and seeds are recorded in `evaluation.json`.

Each split contains 90 four-second mono waveforms at 2048 Hz. Scaling is fitted
only on training features. The signal model ties class to carrier and modulation
bands; its score does not estimate performance on field recordings or establish
tonnage from acoustics. Use labeled recordings and vessel-disjoint splits before
interpreting classes physically. Frequency shifts model Doppler effects, without
estimating radial velocity. Waveforms must contain a nonzero finite signal.
MIT.

`--waveform` accepts a one-dimensional NumPy array and emits `centroid_hz`,
`rolloff_hz`, and `blade_rate_hz` as JSON. Pickled arrays are rejected. This path
extracts features from your recording; it does not apply the generated-signal
classifier to an unvalidated recording domain. Omit the option to run evaluation.
For a mono recording saved as `propeller.npy`, run:

```sh
sonar-classify --waveform propeller.npy --sample-rate 2048
```

## Development

```sh
git clone https://github.com/melgoldstein/acoustic-sonar-doppler-fft.git
cd acoustic-sonar-doppler-fft
python -m venv .venv
# Activate .venv with your shell, then:
python -m pip install -e ".[test]" ruff==0.17.3 build
python -m pytest -q
python -m ruff check --select E9,F63,F7,F82 src tests
python -m ruff format --check src tests
python -m build
```

Report defects with the command, input shape or minimal fixture, and runtime version.
Regression tests should reproduce the failure and preserve the documented units.
