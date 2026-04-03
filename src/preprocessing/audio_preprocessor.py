import numpy as np
import librosa
import noisereduce as nr

class PreProcessingPipeline:
    def __init__(self, audiofile, sr):
        self.audiofile = audiofile
        self.sr = sr
    
    def process(self, filepath):
        audio = librosa.load(filepath,sr=self.sr)

        audio_trim, idx = librosa.effects.trim(audio)
        audio_trim_nr = nr.reduce_noise(audio_trim, prop_decrease=0.5)
        audio_processed = None

        metadata = {"Shape" : None}


        