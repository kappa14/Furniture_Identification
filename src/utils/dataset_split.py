import shutil
import os
import numpy as np
import argparse


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)


def main(path_to_data, path_to_test_data, split_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))

    # calculates how many train train_data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - split_ratio))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_test_data, dirs[i])

        # creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves train_data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Split Dataset")
    parser.add_argument("--data_path", required=True,
                        help="Path to train_data")
    parser.add_argument("--test_data_path", required=True,
                        help="Path to test train_data")
    parser.add_argument("--split_ratio", required=True,
                        help="Split ratio")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_path, args.test_data_path, float(args.split_ratio))


# Following command splits the dataset into train/val subsets :
#
# python dataset_split.py --data_path=/path1 --test_data_path=/path2 --split_ratio=0.8