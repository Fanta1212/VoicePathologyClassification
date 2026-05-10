import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt

#######################
## PREPROCESSING
#######################

def silence_trim(audio_arr: np.ndarray, top_db: int) -> np.ndarray:
    trimmed, tidx = librosa.effects.trim(audio_arr, top_db=top_db)

    if tidx[1] == len(audio_arr) or len(trimmed) == 0:
        return audio_arr
    return trimmed 

def reduce_noise(audio_arr: np.ndarray, sample_rate: int, prop_decrease: float) -> np.ndarray:
    clean_audio_arr = np.nan_to_num(audio_arr,nan=0,posinf=0,neginf=0)

    if np.max(np.abs(clean_audio_arr)) < 1e-4: # silence check
        return clean_audio_arr

    nr_audio_arr = nr.reduce_noise(clean_audio_arr,sr=sample_rate,prop_decrease=prop_decrease)
    return np.nan_to_num(nr_audio_arr,nan=0,posinf=0,neginf=0)

def peak_normalization(audio_arr: np.ndarray) -> np.ndarray:
    audio_arr = np.nan_to_num(audio_arr,nan=0,posinf=0,neginf=0)

    peak = np.max(np.abs(audio_arr))

    if peak < 1e-4:     # avoid division by 0
        return audio_arr
    return audio_arr / peak

def resample_to_16kHz(audio_arr: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio_arr
    return librosa.resample(audio_arr, orig_sr=sample_rate, target_sr=16000)

#################################################
## FEATURE EXTRACTION
#################################################

def extract_f0(audio_arr: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extracts fundamental frequency (F0) contour using probabilistic YIN algorithm.
    """

    f0, _, _ = librosa.pyin(audio_arr, 
                            sr = sample_rate, 
                            fmin = librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7')
                            )
    return f0

def extract_local_jitter(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Computes the approximate local jitter of an audio signal.
    """

    f0 = extract_f0(audio_arr, sample_rate)

    f0 = f0[(~np.isnan(f0)) & (f0 > 0)]

    periods = 1/f0

    if len(periods) < 2:
        return np.nan

    difference = np.abs(np.diff(periods))

    return np.mean(difference) / np.mean(periods)

def extract_rap_jitter(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Computes the approximate RAP (Relative Average Perturbation) Jitter of an audio signal.
    """

    f0 = extract_f0(audio_arr, sample_rate)

    f0 = f0[(~np.isnan(f0)) & (f0 > 0)]

    periods = 1/f0
    rap_terms = []

    if len(periods) < 3:
        return np.nan

    for i in range(1, len(periods) - 1):
        avg = (periods[i-1] + periods[i] + periods[i+1]) / 3
        rap_terms.append(np.abs(periods[i] - avg))

    return np.mean(rap_terms) / np.mean(periods)

def extract_ppq5_jitter(audio_arr: np.ndarray, sample_rate: int) -> float:
    """
    Computes the approximate PPQ5 (Pitch Perturbation Quotient, 5-Points) Jitter of an audio signal.
    """
    f0 = extract_f0(audio_arr, sample_rate)

    f0 = f0[(~np.isnan(f0)) & (f0 > 0)]

    periods = 1/f0
    ppq5_terms = []

    if len(periods) < 5:
        return np.nan
    
    for i in range(2, len(periods) - 2):
        avg = (periods[i-2] + periods[i-1] + periods[i] + periods[i+1] + periods[i+2]) / 5
        ppq5_terms.append(np.abs(periods[i] - avg))
    
    return np.mean(ppq5_terms) / np.mean(periods)

#################################################

def plot_waveform(audio_arr: np.ndarray, sample_rate: int):
    t = np.linspace(0, len(audio_arr)/sample_rate,num=len(audio_arr))
    
    plt.figure(figsize=(15,4))
    plt.plot(t, audio_arr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def plot_spectogram(audio_arr: np.ndarray,sample_rate: int):
    audio_stft = librosa.stft(audio_arr)
    audio_db = librosa.amplitude_to_db(audio_stft)
    
    plt.figure(figsize=(15,4))
    librosa.display.specshow(audio_db,sr=sample_rate,x_axis="time",y_axis="hz",cmap="magma")
    plt.title("Spectogram")
    plt.colorbar(format="%2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    