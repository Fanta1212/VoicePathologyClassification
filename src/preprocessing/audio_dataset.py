import numpy as np
import librosa
import noisereduce as nr
from src.utils.utils import *
import os
import pathlib

directoryPath = str(pathlib.Path.home()) + os.sep + "VoicePathologyClassification" + os.sep + "VoicePathologyClassification" + os.sep + "data"

class Preprocess:
    def __init__(self, audio_filename: str, prop_decrease: float, top_db: int):
        self.audio_filename = audio_filename
        self.prop_decrease = prop_decrease
        self.top_db = top_db


    def process(self):
        audio_arr, sample_rate = librosa.load(directoryPath + os.sep + "raw" + os.sep + self.audio_filename,sr=None)

        trimmed_audio = silence_trim(audio_arr, top_db=self.top_db)
        noise_reduced_audio = reduce_noise(trimmed_audio,sample_rate=sample_rate, prop_decrease=self.prop_decrease)    
        resampled_audio = resample_to_16kHz(noise_reduced_audio, sample_rate=sample_rate)
        normalized_audio = peak_normalization(resampled_audio) 

        metadata = {"Filename" : self.audio_filename,
                    "Duration" : round(len(normalized_audio)/16000, 4),
                    "Sample Rate" : 16000}
        
        return normalized_audio, metadata


if __name__ == "__main__":
    x = Preprocess(r"halitVoice.wav",0.5,40)
    voice_resampled, md1= x.process()
    print(md1)

    y = Preprocess(r"5 Seconds of silence.wav",0.5,40)
    voice_processed, md2 = y.process()
    print(md2)


        