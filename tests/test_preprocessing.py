import numpy as np
import os
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.audio_preprocessor import PreProcessingPipeline

def test_trim_on_silence_test():
    audio = PreProcessingPipeline(r"C:\Users\ernun\VoicePathologyClassification\VoicePathologyClassification\data\raw\5 seconds of silence - Ira Bnut (128k).wav",0)
    
    try:
        processed, _ = audio.process()
    except Exception:
        pytest.fail("Process() crashed on silent audio: ", Exception)

def test_output_sample_rate_16kHz():
    audio = PreProcessingPipeline(r"C:\Users\ernun\VoicePathologyClassification\VoicePathologyClassification\data\raw\voice001.wav",0)

    processed, metadata = audio.process()

    assert (metadata["Sampling Rate"] == 16000), (
        "Sampling rate is not 16kHz."
    )

def test_output_amplitude():
    audio = audio = PreProcessingPipeline(r"C:\Users\ernun\VoicePathologyClassification\VoicePathologyClassification\data\raw\voice001.wav",0)
    
    processed, _ = audio.process()
    assert (np.max(processed) <= 1) and (np.min(processed) >= -1), (
        "The audio signal is not normalized."
    )



