import argparse
import os
import subprocess

import numpy as np
import pandas as pd

log_file = "processed_datasets.log"


def get_dataset_info(dataset_path):
    if dataset_path.endswith(".csv"):
        # Read datas from a CSV file
        data = pd.read_csv(dataset_path, header=None)
        num_samples, num_features = data.shape
    elif dataset_path.endswith(".npz"):
        # Read datas from a NPZ file
        with np.load(dataset_path) as npzfile:
            key = list(npzfile.keys())[0]
            data = npzfile[key]
            num_samples, num_features = data.shape
    else:
        raise ValueError(
            "unsuported dataset file type : should be .csv of .npz"
        )

    return num_samples, num_features


def calculate_metric(num_samples, num_features):
    return num_samples * (num_features ** (1 / 4))


def run_simulation(
    dataset_path,
    perform_selective_noise,
    perform_shapley_values,
    perform_perc_keep,
):
    command = f'python dte_itered.py -d"{dataset_path}"'
    if perform_selective_noise:
        command += " -n"
    if perform_shapley_values:
        command += " -s"
    if perform_perc_keep:
        command += " -p"

    print(f"Executing command: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Attendre que le processus actuel se termine avant de lancer le suivant


def load_processed_datasets():
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            processed_datasets = f.read().splitlines()
    else:
        processed_datasets = []
    return processed_datasets


def save_processed_dataset(dataset_path):
    with open(log_file, "a") as f:
        f.write(dataset_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_simulation",
        description="This program perform dte_itered on all the dataset present in a given folder",
    )
    parser.add_argument(
        "-f",
        "--data_folder",
        default="./data",
        help="folder containing the dataset to compute",
    )
    parser.add_argument(
        "-n",
        "--selective_noise",
        help="calculate the selective noise",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--shapley_values",
        help="calculate the shapley values",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--perc_keep",
        help="Calculate F1 for different percentage_keep values",
        action="store_true",
    )
    args = parser.parse_args()
    data_folder = args.data_folder

    dataset_paths = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if (f.endswith(".csv") or f.endswith(".npz"))
    ]

    # loading already processed dataset
    processed_datasets = load_processed_datasets()

    # filter not processed dataset
    dataset_paths = [
        path for path in dataset_paths if path not in processed_datasets
    ]
    dataset_paths = dataset_paths[:1]
    # Sorting dataset by their size (size computed as num_sample*(num_features**(1/4)))
    dataset_info = []
    for dataset_path in dataset_paths:
        num_samples, num_features = get_dataset_info(dataset_path)
        metric = calculate_metric(num_samples, num_features)
        dataset_info.append((metric, dataset_path))
    dataset_info.sort(key=lambda x: x[0])

    # Executing simulation on the dataset, starting by the smaller
    for _, dataset_path in dataset_info:
        if dataset_path in processed_datasets:
            continue  # Skip already processed datasets

        run_simulation(
            dataset_path,
            args.selective_noise,
            args.shapley_values,
            args.perc_keep,
        )
        save_processed_dataset(dataset_path)
