import numpy as np
import pytest
from src.preprocessing import Preprocess

def test_trim_on_silence_test():
    audio = Preprocess(r"5 seconds of silence.wav",0,60)
    
    try:
        processed, _ = audio.process()
    except Exception:
        pytest.fail("Process() crashed on silent audio: ", Exception)

def test_output_sample_rate_16kHz():
    audio = Preprocess(r"148-phrase.wav",0,60)

    processed, metadata = audio.process()

    assert (metadata["Sample Rate"] == 16000), (
        "Sample rate is not 16kHz."
    )

def test_output_amplitude():
    audio = audio = Preprocess(r"148-phrase.wav",0,60)
    
    processed, _ = audio.process()
    assert (np.max(processed) <= 1) and (np.min(processed) >= -1), (
        "The audio signal is not normalized."
    )



if __name__ == "__main__" :
    print("-"*50)
    print("UNIT TEST")
    print("-"*50)

    test_trim_on_silence_test()
    test_output_sample_rate_16kHz()
    test_output_amplitude()
    print("-"*50)
