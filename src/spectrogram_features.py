import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert


def spectral_features(
    samples: NDArray[np.float64], sample_rate: float
) -> NDArray[np.float64]:
    waveform = np.asarray(samples, dtype=float)
    if (
        waveform.ndim != 1
        or len(waveform) < 128
        or not np.isfinite(waveform).all()
        or not np.isfinite(sample_rate)
        or sample_rate <= 100
    ):
        raise ValueError("finite mono waveform and sample rate above 100 Hz required")
    centered = waveform - waveform.mean()
    if np.linalg.norm(centered) < 1e-10:
        raise ValueError("silent waveform")
    window = np.hanning(len(centered))
    power = np.abs(np.fft.rfft(centered * window)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), 1 / sample_rate)
    centroid = float(power @ frequencies / power.sum())
    rolloff = float(frequencies[np.searchsorted(np.cumsum(power), 0.85 * power.sum())])
    envelope = np.abs(hilbert(centered))
    modulation = np.abs(np.fft.rfft((envelope - envelope.mean()) * window)) ** 2
    blade_band = (frequencies >= 2) & (frequencies <= 40)
    blade_rate = float(frequencies[blade_band][np.argmax(modulation[blade_band])])
    return np.array([centroid, rolloff, blade_rate])
