import numpy as np
import pytest
from sonar_dsp.spectrogram_features import spectral_features
from sonar_dsp.classifier import evaluate


def test_carrier_and_modulation():
    timeline = np.arange(8192) / 2048
    waveform = (1 + 0.4 * np.sin(2 * np.pi * 12 * timeline)) * np.sin(
        2 * np.pi * 300 * timeline
    )
    centroid, rolloff, blade_rate = spectral_features(waveform, 2048)
    assert centroid == pytest.approx(300, abs=1)
    assert rolloff == pytest.approx(300, abs=15)
    assert blade_rate == pytest.approx(12, abs=0.25)
    np.testing.assert_allclose(
        spectral_features(waveform * 3 + 8, 2048), [centroid, rolloff, blade_rate]
    )


@pytest.mark.parametrize("waveform", [np.zeros(128), np.ones(10), np.full(256, np.nan)])
def test_invalid(waveform):
    with pytest.raises(ValueError):
        spectral_features(waveform, 2048)


def test_holdout():
    for result in evaluate()["models"].values():
        assert result["balanced_accuracy"] > 0.9
