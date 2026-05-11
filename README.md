# VoicePathologyClassification

### 30.03.2026

### MOTIVATION
Roughly 1 in 13 adults get affected by voice disorders annually. The procedures that are applied to properly diagnose voice disorders are mostly invasive and uncomfortable, as well as being expensive. Although there are acoustic analysis tools that are used in researches, they are trained mostly on a single language such as English or German, making them unusable in different languages.

The aim of this project is to construct an open-source voice analysis tool to help in the diagnosis of voice pathologies using a database created with Turkish voice records that me and my team will build.

## PREPROCESSING PIPELINE / BATCH PREPROCESSING

Each audio is preprocessed in 4 steps: Trimming the audio from the start and the end according to the "top_db" argument, reducing the unnecessary noise and the background noise according to the "prop_decrease" argument, resampling the audio to 16 kHz because it captures everything that is relevant in human voice (up to 8kHz; with the Nyquist Theorem, 16kHz), and finally normalizing it by the maximum amplitude of the audio signal to have the audio signals at the same range [-1,1]. 


## ACOUSTIC FEATURES
### F0 (Fundamental Frequency)
Every 20-40 ms of the audio signal is framed and the F0 is calculated with the autocorrelation method in every single frame, this gives the F0 frequency overtime which is the pitch in the frame.

### Jitter
The change in the fundamental frequency (pitch period) from one cycle to the next.

#### Local Jitter
Detects very small changes in the pitch period, robustness is low.

#### RAP Jitter
Is not very sensitive to the small changes in the audio, but more robust than Local Jitter.

#### PPQ5 Jitter
Very low sensitivity to the changes in the audio signal, but very robust.

