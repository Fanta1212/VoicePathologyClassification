import librosa
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt


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
    return librosa.resample(audio_arr, orig_sr=sample_rate, target_sr=16000)

def waveform(audio_arr: np.ndarray, sample_rate: int):
    t = np.linspace(0, len(audio_arr)/sample_rate,num=len(audio_arr))
    
    plt.figure(figsize=(15,4))
    plt.plot(t, audio_arr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def spectogram(audio_arr: np.ndarray,sample_rate: int):
    audio_stft = librosa.stft(audio_arr)
    audio_db = librosa.amplitude_to_db(audio_stft)
    
    plt.figure(figsize=(15,4))
    librosa.display.specshow(audio_db,sr=sample_rate,x_axis="time",y_axis="hz",cmap="magma")
    plt.title("Spectogram")
    plt.colorbar(format="%2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    