import numpy as np
from scipy import signal
import os

import matplotlib.pyplot as plt

data_dir = "recorded_signals"
fs = 1000  # sampling frequency 1000Hz
window_length = 200  # ms

gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_index', 'pinch_middle',
                    'pinch_ring', 'pinch_small']

output_data = []

for session_dir in os.listdir(data_dir):
    session_dir_path = os.path.join(data_dir, session_dir)

    for g in gestures_classes:
        file_name = f"{g}.csv"

        # Load signals
        loaded_signal = np.loadtxt(os.path.join(session_dir_path, file_name), delimiter=',')

        # Highpass filter
        fd = 10  # Hz
        n_fd = fd / (fs / 2)  # normalized frequency
        b, a = signal.butter(1, n_fd, 'highpass')
        hp_filtered_signal = signal.lfilter(b, a, loaded_signal.T)

        # Notch filter
        notch_filtered_signal = hp_filtered_signal  # cut off the beginning and transpose
        for f0 in [50, 100, 200]:  # 50Hz and 100Hz notch filter
            Q = 5  # quality factor
            b, a = signal.iirnotch(f0, Q, fs)
            notch_filtered_signal = signal.lfilter(b, a, notch_filtered_signal)

        # Transpose back to shape=(samples, channels)
        filtered_signal = notch_filtered_signal.T

        # Calculate new dimensions
        n_windows = filtered_signal.shape[0] // window_length
        n_channels = filtered_signal.shape[1]

        # Reshape by windows
        windowed_signal = np.resize(filtered_signal, (n_windows, window_length, n_channels))

        # Cut the first two windows - filter response has to be fixed
        # In an online preprocessing use np.concatenate((data,)*3, axis=0) before filtering and after it
        # reject the first and the second copy of the signal
        # 200ms window X3 => 600ms (three copies) => filtering => reject first 400ms => you've got filtered data
        final_windowed_signal = windowed_signal[2:]

        #  TODO save to the file or add to list, check out .npz (writing once vs appending) in put_emg saving