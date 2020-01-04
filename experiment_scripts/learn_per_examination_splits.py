import os
import os.path as osp

from pegc.training import train_loop
from pegc.training.utils import load_json

EXAMINATIONS_PREPARED_TRAIN_TEST_SPLITS_DIR_PATH = '/media/ja/CCTV_nagrania/mkm_archive/put_emg/data/raw_filtered_data_subjects_split_window_size_1024_window_stride_512_cv_splits_standarized'
RESULTS_OUTPUT_DIR_PATH = '../sandbox/learn_per_examination_splits_output'
CONFIG_TEMPLATE_PATH = './config_template.json'


def run_experiment(examinations_prepared_train_test_splits_dir_path: str,
                   results_output_dir_path: str, training_config_file_path: str) -> None:
    # Note: training config file is a dict in json format, while it's content should be arguments which will override
    # some of (or all) arguments of the train loop function.
    # Template with "default" train loop arguments/what can be changed is included in experiments scripts dir.
    possible_splits = {'split_0': {'train': ('sequential', 'repeats_short'), 'test': 'repeats_long'},
                       'split_1': {'train': ('sequential', 'repeats_long'), 'test': 'repeats_short'},
                       'split_2': {'train': ('repeats_short', 'repeats_long'), 'test': 'sequential'}}
    training_config = load_json(training_config_file_path)

    os.makedirs(results_output_dir_path, exist_ok=True)
    for examination_dir in os.listdir(examinations_prepared_train_test_splits_dir_path):
        examination_dir_path = osp.join(examinations_prepared_train_test_splits_dir_path, examination_dir)
        examination_id = examination_dir

        results_examination_dir_path = osp.join(results_output_dir_path, examination_dir)
        os.makedirs(results_examination_dir_path, exist_ok=True)
        for split_dir in os.listdir(examination_dir_path):
            split_dir_path = osp.join(examination_dir_path, split_dir)
            results_split_dir_path = osp.join(results_examination_dir_path, split_dir)
            train_loop(split_dir_path, results_split_dir_path, **training_config)
        # break

    # TODO: aggregate results, plots


if __name__ == '__main__':
    run_experiment(EXAMINATIONS_PREPARED_TRAIN_TEST_SPLITS_DIR_PATH, RESULTS_OUTPUT_DIR_PATH, CONFIG_TEMPLATE_PATH)
