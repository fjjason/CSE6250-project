import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mne
from mne.io import read_raw_edf
import os
import re
import scipy
from scipy.signal import welch
from scipy.integrate import simps
from time import time
import pickle
import matplotlib.cm as cm

# Reference: https://raphaelvallat.com/bandpower.html
# Reference: https://stackoverflow.com/questions/44547669/python-numpy-equivalent-of-bandpower-from-matlab
def bandpower(seq, sampling_frequency, frequency_band):
    f, Pxx = scipy.signal.periodogram(seq, fs=sampling_frequency)
    ind_min = scipy.argmax(f > frequency_band[0]) - 1
    ind_max = scipy.argmax(f > frequency_band[1]) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

eeg_info_df = pd.read_csv('data/eeg_info_df.csv')

frequency_bands = {'Beta': [12, 30],
                   'Alpha': [8, 12],
                   'Theta': [4, 8],
                   'Delta': [0.5, 4]}
band_names = list(frequency_bands.keys())
shared_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']

feature_list = []
label_list = []
path_list = []
sf = 100

time_interval = 3600

tic = time()
for i, path in enumerate(eeg_info_df.path.values):
    raw_edf = read_raw_edf(os.path.join('data', eeg_info_df.path.iloc[i]))
    edf_df = raw_edf.to_data_frame()
    edf_df = edf_df[edf_df['EEG Fpz-Cz'] > -1000] # remove outliers

    n = edf_df.shape[0]
    n_hours = n//(time_interval*100)
    for j in range(n_hours):
        temp_df = edf_df.iloc[(j*time_interval*100):((j+1)*time_interval*100), :]
        features = [bandpower(temp_df[channel].values, sf, frequency_bands[band])
                    for channel in shared_channels for band in band_names]
        feature_list.append(features)
        label_list.append(1 if path.find('telemetry') > -1 else 0)
        path_list.append(path)
    if i == 0 : print('1st eeg takes {} seconds\n'.format(round(time()-tic)))

toc = time()
print('one hour band power calculation for 4 bands x 4 channels takes {} seconds'.format(round(toc-tic)))
# one hour band power calculation for 4 bands x 4 channels takes 897 seconds

y = np.array(label_list)
X = np.array(feature_list)
paths = np.array(path_list)

pickle.dump((X, y, paths), open('data/model_data_v1.sav', 'wb'))

