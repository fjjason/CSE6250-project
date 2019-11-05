import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mne
from mne.io import read_raw_edf
import os
import re
import seaborn as sns
sns.set()

sample_data = read_raw_edf('data/sleep-cassette/SC4041E0-PSG.edf')
channel_names = sample_data.info['ch_names']
Fs = sample_data.info['sfreq']
eeg_waveform = sample_data.to_data_frame()

# taking average of every 100 seconds and show first channel time series plot
wave1 = eeg_waveform.iloc[:, 0].values.reshape(-1, 10000)
pd.Series(wave1.mean(axis=1)).plot()

def plot_psg(eeg, save_path):
    eeg_waveform = eeg.to_data_frame()
    channels = eeg_waveform.columns
    plt.figure(figsize=[10, 10])
    for i, channel in enumerate(channels):
        plt.subplot(3, 3, i+1)
        wave = eeg_waveform.iloc[:, i].values.reshape(-1, 10000)
        plt.plot(wave.mean(axis=1), color='k')
        plt.title(channel)
        plt.xlabel('time (10s)')
    plt.tight_layout()
    plt.savefig(save_path)

cassete_sample = read_raw_edf('data/sleep-cassette/SC4041E0-PSG.edf')
plot_psg(cassete_sample, 'plot/SC4041E0-PSG.png')

telemetry_sample = read_raw_edf('data/sleep-telemetry/ST7011J0-PSG.edf')
plot_psg(telemetry_sample, 'plot/ST7011J0-PSG.png')
# the final drop appears to be outlier

psg_paths = pd.read_table('data/RECORDS', header=None).iloc[:, 0].values

eeg_info_list = []
for psg_path in psg_paths:
    raw_edf = read_raw_edf(os.path.join('data', psg_path))
    edf_df = raw_edf.to_data_frame()
    eeg_info_list.append([psg_path, edf_df.columns.tolist(), edf_df.shape[0]])

eeg_info_df = pd.DataFrame.from_records(eeg_info_list, columns=['path', 'channels', 'n_records'])
eeg_info_df.to_csv('data/eeg_info_df.csv', index=False)

eeg_info_df['n_channels'] = eeg_info_df.channels.map(len)
eeg_info_df['label'] = np.where(eeg_info_df.path.map(lambda x: x.split('/')[0])=='sleep-telemetry', 1, 0)
eeg_info_df['label'].value_counts()
# 0    153
# 1     44

np.unique(eeg_info_df[['label', 'n_channels']].values, axis=0)
# array([[0, 7],
#        [1, 5]])

channels0 = eeg_info_df.channels.values[0]
channels1 = eeg_info_df.channels.values[-1]
[i for i in channels1 if i in channels0]
# ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
