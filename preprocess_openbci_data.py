import numpy as np
from scipy import signal
import os
import click
from tqdm import tqdm


def filter_signals(s: np.ndarray, fs: int) -> np.ndarray:
    # Highpass filter
    fd = 10  # Hz
    n_fd = fd / (fs / 2)  # normalized frequency
    b, a = signal.butter(1, n_fd, 'highpass')
    hp_filtered_signal = signal.lfilter(b, a, s.T)

    # Notch filter
    notch_filtered_signal = hp_filtered_signal  # cut off the beginning and transpose
    for f0 in [50, 100, 200]:  # 50Hz and 100Hz notch filter
        Q = 5  # quality factor
        b, a = signal.iirnotch(f0, Q, fs)
        notch_filtered_signal = signal.lfilter(b, a, notch_filtered_signal)

    return notch_filtered_signal.T


@click.command()
@click.argument('data_dir_path', type=click.Path(file_okay=False, exists=True))
@click.argument('session_dir', type=str)
@click.argument('preprocessed_data_dir_path', type=click.Path(file_okay=False))
def preprocess_openbci_data(data_dir_path: str, session_dir: str, preprocessed_data_dir_path: str) -> None:
    fs = 1000  # sampling frequency 1000Hz
    window_length = 200  # ms
    stride = window_length

    gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_index', 'pinch_middle',
                        'pinch_ring', 'pinch_small']

    possible_splits = {'split_0': {'train': ('sequential', 'repeats_short'), 'test': 'repeats_long'},
                       'split_1': {'train': ('sequential', 'repeats_long'), 'test': 'repeats_short'},
                       'split_2': {'train': ('repeats_short', 'repeats_long'), 'test': 'sequential'}}

    # Dict for organize data by trajectories
    session_data = {}

    results_dir = os.path.join(preprocessed_data_dir_path, session_dir)

    for trajectory_dir in tqdm(('repeats_long', 'repeats_short', 'sequential')):
        trajectory_dir_path = os.path.join(data_dir_path, session_dir, trajectory_dir)

        X = []
        y = []

        for g in gestures_classes:
            file_name = f"{g}.csv"

            # Load signals
            loaded_signal = np.loadtxt(os.path.join(trajectory_dir_path, file_name), delimiter=',')

            # Do filtering
            filtered_signal = filter_signals(loaded_signal, fs)

            # Replace 8th channel with 7th channel (it will be two copies of 7th channel)
            filtered_signal[:, 7] = filtered_signal[:, 6]

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

            # Shuffle windows
            # np.random.shuffle(final_windowed_signal)

            # Add data to the lists
            X.append(final_windowed_signal)

            # Add labels (one hot) to the lists
            one_hot_label = np.array([int(g == x) for x in gestures_classes])
            y.append(np.tile(one_hot_label, (final_windowed_signal.shape[0], 1)))

        # Save in session data dict
        session_data[trajectory_dir] = {'X': np.concatenate(X), 'y': np.concatenate(y)}

    # Organize data in splits
    for split_name, train_test_info in possible_splits.items():
        X_train = np.concatenate([session_data[train_test_info['train'][0]]['X'],
                                  session_data[train_test_info['train'][1]]['X']])
        y_train = np.concatenate([session_data[train_test_info['train'][0]]['y'],
                                  session_data[train_test_info['train'][1]]['y']])

        X_test = session_data[train_test_info['test']]['X']
        y_test = session_data[train_test_info['test']]['y']

        # Save data to the files .npz
        cur_split_dst_dir_path = os.path.join(results_dir, split_name)
        os.makedirs(cur_split_dst_dir_path, exist_ok=True)
        np.savez_compressed(os.path.join(cur_split_dst_dir_path, 'X_train.npz'), arr=X_train)
        np.savez_compressed(os.path.join(cur_split_dst_dir_path, 'X_test.npz'), arr=X_test)
        np.savez_compressed(os.path.join(cur_split_dst_dir_path, 'y_train.npz'), arr=y_train)
        np.savez_compressed(os.path.join(cur_split_dst_dir_path, 'y_test.npz'), arr=y_test)


if __name__ == '__main__':
    preprocess_openbci_data()
