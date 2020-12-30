import numpy as np
from scipy import signal
import os
import click


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
@click.argument('preprocessed_data_dir_path', type=click.Path(file_okay=False))
def preprocess_openbci_data(data_dir_path: str, preprocessed_data_dir_path: str, test_size: int = 0.2) -> None:
    fs = 1000  # sampling frequency 1000Hz
    window_length = 200  # ms
    stride = window_length

    gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_index', 'pinch_middle',
                        'pinch_ring', 'pinch_small']

    for session_dir in os.listdir(data_dir_path):
        session_dir_path = os.path.join(data_dir_path, session_dir)

        for split_dir in os.listdir(session_dir_path):
            split_dir_path = os.path.join(session_dir_path, split_dir)

            output_test_X = []
            output_train_X = []
            output_test_y = []
            output_train_y = []

            for g in gestures_classes:
                file_name = f"{g}.csv"

                # Load signals
                loaded_signal = np.loadtxt(os.path.join(split_dir_path, file_name), delimiter=',')

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
                np.random.shuffle(final_windowed_signal)

                # Update number of windows
                n_windows = final_windowed_signal.shape[0]

                # Add data to the lists
                output_train_X.append(final_windowed_signal[int(n_windows*test_size):])
                output_test_X.append(final_windowed_signal[:int(n_windows*test_size)])

                # Add labels (one hot) to the lists
                one_hot_label = np.array([int(g == x) for x in gestures_classes])
                n_train_windows = output_train_X[-1].shape[0]
                n_test_windows = output_test_X[-1].shape[0]
                output_train_y.append(np.tile(one_hot_label, (n_train_windows, 1)))
                output_test_y.append(np.tile(one_hot_label, (n_test_windows, 1)))

            # Concatenate data in lists and save it to the files .npz
            results_dir_path = os.path.join(preprocessed_data_dir_path, session_dir, split_dir)
            os.makedirs(results_dir_path)
            np.savez_compressed(os.path.join(results_dir_path, 'X_train.npz'), arr=np.concatenate(output_train_X))
            np.savez_compressed(os.path.join(results_dir_path, 'X_test.npz'), arr=np.concatenate(output_test_X))
            np.savez_compressed(os.path.join(results_dir_path, 'y_train.npz'), arr=np.concatenate(output_train_y))
            np.savez_compressed(os.path.join(results_dir_path, 'y_test.npz'), arr=np.concatenate(output_test_y))


if __name__ == '__main__':
    preprocess_openbci_data()
