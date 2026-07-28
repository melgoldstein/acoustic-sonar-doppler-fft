import argparse
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from .spectrogram_features import spectral_features


def acoustic_trials(seed: int, count: int = 30) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    timeline = np.arange(8192) / 2048
    features, labels = [], []
    for category, (carrier, blade_rate) in enumerate(((500, 25), (300, 15), (140, 7))):
        for _ in range(count):
            doppler_factor = generator.uniform(0.94, 1.06)
            frequency = (carrier + generator.normal(0, 25)) * doppler_factor
            modulation = (blade_rate + generator.normal(0, 1)) * doppler_factor
            phase = generator.uniform(0, 2 * np.pi)
            waveform = (1 + 0.4 * np.sin(2 * np.pi * modulation * timeline)) * np.sin(
                2 * np.pi * frequency * timeline + phase
            )
            waveform += generator.normal(0, generator.uniform(0.1, 0.4), len(timeline))
            features.append(spectral_features(waveform, 2048))
            labels.append(category)
    return np.array(features), np.array(labels)


def evaluate() -> dict[str, object]:
    training, training_labels = acoustic_trials(31)
    holdout, holdout_labels = acoustic_trials(87)
    results = {}
    for name, estimator in (
        ("logistic", LogisticRegression(max_iter=1000)),
        ("svc", SVC()),
    ):
        model = make_pipeline(StandardScaler(), estimator).fit(
            training, training_labels
        )
        predictions = model.predict(holdout)
        results[name] = {
            "balanced_accuracy": float(
                balanced_accuracy_score(holdout_labels, predictions)
            ),
            "confusion_matrix": confusion_matrix(holdout_labels, predictions).tolist(),
        }
    return {
        "train_seed": 31,
        "holdout_seed": 87,
        "classes": ["light", "medium", "heavy"],
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate spectral vessel-class features"
    )
    parser.parse_args()
    print(json.dumps(evaluate(), indent=2))


if __name__ == "__main__":
    main()
