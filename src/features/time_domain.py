import librosa
import numpy as np
from scipy.signal import find_peaks

# Fundamental Frequency 
def extract_f0(audio_arr: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extracts fundamental frequency (F0) contour using probabilistic YIN algorithm.
    """

    f0, voiced_flag, _ = librosa.pyin(audio_arr, 
                            sr = sample_rate, 
                            fmin = librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7')
                            )
    return f0, voiced_flag

# Local Jitter, RAP Jitter, PPQ5 Jitter
# implement local jitter
def jitter_local(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Compute local jitter from an F0 contour.

    Captures the average between consecutive pitch periods,
    expressed as a percentage.

    Definition
    -----------
    Given N valid pitch periods T[0], ...,T[N-1]:

        local_jitter (%) = 100 x mean(|T[i] - T [i - 1]|, i = 0...N-1)
                                _______________________________________
                                                mean(T)
                                
    """
    f0, voiced_flag = extract_f0(audio_arr, sample_rate)
    f0 = f0[voiced_flag]

    periods = 1/f0

    diffs = np.abs(np.diff(periods))
    avg = np.mean(periods)

    return (np.mean(diffs) / avg)*100

# implement RAP jitter
def jitter_rap(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Compute RAP (Relative Average Perturbation) jitter from an F0 contour.

    Captures the average between 3 consecutive pitch periods,
    expressed as a percentage.

    Definiton
    ----------
    Given N valid pitch periods T[0],...,T[N-1]

        rap_jitter (%) = 100 x mean(|T[i] - (T[i-1] + T[i] + T[i+1]) / 3, i = 1,...,N-2|)
                                ____________________________________________________________
                                                            mean(T)

    """
    f0, voiced_flag = extract_f0(audio_arr, sample_rate)
    f0 = f0[voiced_flag]

    t = 1/f0

    diffs = []
    for i in range(1, len(t) - 1):
        avg = (t[i-1] + t[i] + t[i+1]) / 3
        diff = np.abs(t[i] - avg)
        diffs.append(diff)
    return (np.mean(diffs) / np.mean(t))*100

# implement PPQ5 jitter
def jitter_ppq5(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Compute PPQ5 (Period Perturbation Quotient, 5-Point) jitter from an F0 contour.

    Captures the average between 5 consecutive pitch periods,
    expressed as a percentage.

    Definiton
    ----------
    Given N valid pitch periods T[0],...,T[N-1]

        ppq5_jitter (%) = 100 x mean(|T[i] - (T[i-2] + T[i-1] + T[i] + T[i+1] + T[i+2]) / 5, i = 2,...,N-3|)
                                _____________________________________________________________________________
                                                                        mean(T)

    """
    f0, voiced_flag = extract_f0(audio_arr, sample_rate)
    f0 = f0[voiced_flag]

    t = 1/f0

    diffs = []
    for i in range(2, len(t) - 2):
        avg = (t[i-2] + t[i-1] + t[i] + t[i+1] + t[i+2]) / 5
        diff = np.abs(t[i] - avg)
        diffs.append(diff)
    return (np.mean(diffs) / np.mean(t))*100

# Local Shimmer, APQ3 Shimmer, APQ5 Shimmer
def extract_local_shimmer(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Computes the approximate local shimmer of an audio signal.
    """
    f0, voiced_flag, _ = librosa.pyin(audio_arr, 
                                      sr=sample_rate, 
                                      fmin = librosa.note_to_hz('C2'),
                                      fmax=librosa.note_to_hz('C7')
                                      )
    voiced = f0[voiced_flag]
    if len(voiced) == 0:
        return np.nan
    median_f0 = np.nanmedian(voiced)
    min_distance = int(sample_rate / (median_f0*1.5))
    max_distance = int(sample_rate / (median_f0*0.5))
    peaks, _ = find_peaks(np.abs(audio_arr), distance=min_distance)
    peak_distances = np.abs(np.diff(peaks))

    amplitudes = np.abs(audio_arr[peaks])

    if len(amplitudes) < 2:
        return np.nan

    valid_pairs = (peak_distances <= max_distance)

    diff = np.abs(np.diff(amplitudes))[valid_pairs]
    amplitudes = amplitudes[:-1][valid_pairs]

    return np.mean(diff) / np.mean(amplitudes)

