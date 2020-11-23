import os
import pandas as pd
import librosa

root_path = '/Users/saurabh/PycharmProjects/IDS/final_project/music-genre-classification/data/2018655052_small'
tracks_dict = {}
for path, subdirs, files in os.walk(root_path):
    for name in files:
        if name.split('.')[0].isdigit():
            tracks_dict[name.split('.')[0]] = os.path.join(path, name)
print(tracks_dict)
tracks_data = pd.read_csv('./data/fma_metadata/raw_tracks.csv', low_memory=False)
print(tracks_data.head())

tracks_data = tracks_data[tracks_data['track_id'].isin(tracks_dict.keys())]
print(tracks_data.head())
tracks_data = tracks_data[['track_id']]
print(len(tracks_data))
tracks_data.to_csv('./data/tracks.csv')



{'133297': '/Users/saurabh/PycharmProjects/IDS/final_project/music-genre-classification/data/2018655052_small/133/133297.mp3', '099134': '/Users/saurabh/PycharmProjects/IDS/final_project/music-genre-classification/data/2018655052_small/099/099134.mp3', '108925': '/Users/saurabh/PycharmProjects/IDS/final_project/music-genre-classification/data/2018655052_small/108/108925.mp3'}

# import numpy as np
# import librosa
#
#
# def feature_extract(file):
#     """
#     Define function that takes in a file an returns features in an array
#     """
#
#     # get wave representation
#     y, sr = librosa.load(file)
#
#     # determine if instruemnt is harmonic or percussive by comparing means
#     y_harmonic, y_percussive = librosa.effects.hpss(y)
#     if np.mean(y_harmonic) > np.mean(y_percussive):
#         harmonic = 1
#     else:
#         harmonic = 0
#
#     # Mel-frequency cepstral coefficients (MFCCs)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     # temporal averaging
#     mfcc = np.mean(mfcc, axis=1)
#
#     # get the mel-scaled spectrogram
#     spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
#     # temporally average spectrogram
#     spectrogram = np.mean(spectrogram, axis=1)
#
#     # compute chroma energy
#     chroma = librosa.feature.chroma_cens(y=y, sr=sr)
#     # temporally average chroma
#     chroma = np.mean(chroma, axis=1)
#
#     # compute spectral contrast
#     contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     contrast = np.mean(contrast, axis=1)
#
#     return [harmonic, mfcc, spectrogram, chroma, contrast]