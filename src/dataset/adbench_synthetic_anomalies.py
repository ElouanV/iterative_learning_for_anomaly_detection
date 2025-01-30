"""
File from ADBench https://github.com/Minqi824/ADBench.git
that was modified to integrate the new semi-supervised setting
and add more flexibility.
Copyright (c) 2022, Mickey (Minqi)
All rights reserved.
"""
import copy
import logging
import os
import random
from math import ceil

import yaml

import hydra
import numpy as np
import omegaconf
import pkg_resources
from adbench.myutils import Utils
from bidict import bidict
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from src.dataset.generate_synthetic_data_bis import generate_anomalies_scheme

# Path to adbench datasets
DATA_PATH = pkg_resources.resource_filename("adbench", "datasets/")


# currently, data generator only supports for generating the binary classification datasets
class DataGenerator:
    def __init__(
        self,
        seed: int = 42,
        dataset: str = None,
        test_size: float = 0.3,
        generate_duplicates=True,
        n_samples_threshold=1000,
        normal=False,
        config=None,
    ):
        """
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False,
          then datasets with sample size smaller than n_samples_threshold will be dropped
        """

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold
        self.normal = normal
        self.config = config
        # dataset list
        self.dataset_list_classical = [
            os.path.splitext(_)[0]
            for _ in os.listdir(DATA_PATH + "Classical")
            if os.path.splitext(_)[1] == ".npz"
        ]  # classical AD datasets
        self.dataset_list_cv = [
            os.path.splitext(_)[0]
            for _ in os.listdir(DATA_PATH + "CV_by_ResNet18")
            if os.path.splitext(_)[1] == ".npz"
        ]  # CV datasets
        self.dataset_list_nlp = [
            os.path.splitext(_)[0]
            for _ in os.listdir(DATA_PATH + "NLP_by_BERT")
            if os.path.splitext(_)[1] == ".npz"
        ]  # NLP datasets

        # myutils function
        self.utils = Utils()
        self.logger = logging.getLogger(__name__)

    def add_anomalies(self, X_normal, graphical_model, anomalies_scheme: dict, anomalies_nb: dict):
        # Use bidict to store the type of anomaly and an int
        anomaly_code = {
            "cluster": 1,
            "global": 2,
            "additive_noise": 4,
            "multiplicative_noise": 5,
            "local": 6,
        }
        X_anomalies = []
        y_anomalies = []
        explanation = []
        anomaly_code = bidict(anomaly_code)
        anomalies_info = {}
        for anomaly in anomalies_scheme.keys():
            anomalies_info = anomalies_scheme[anomaly]
            if anomaly == "cluster":
                alpha = anomalies_info["alpha"]
                modified_graphical_model = copy.deepcopy(graphical_model)
                modified_graphical_model.means_ = alpha * modified_graphical_model.means_
                anomalies = modified_graphical_model.sample(anomalies_nb[anomaly])
                X_anomalies.append(anomalies)
                y_anomalies.append(np.full(anomalies_nb[anomaly], anomaly_code[anomaly]))
                explanation.append(np.one(anomalies.shape))
                anomalies_info[anomaly] = {"alpha": alpha, "features": "all"}
            elif anomaly == "global":
                selected_features = anomalies_info["features"]
                alpha = anomalies_info["alpha"]
                anomalies = graphical_model.sample(anomalies_nb[anomaly])[0]
                for i in selected_features:
                    low = np.min(X_normal[:, i]) * (alpha)
                    high = np.max(X_normal[:, i]) * (alpha)
                    anomalies[:, i] = np.random.uniform(low, high, anomalies_nb[anomaly])
                X_anomalies.append(anomalies)
                y_anomalies.append(np.full(anomalies_nb[anomaly], anomaly_code[anomaly]))

                exp = np.zeros(anomalies.shape)
                exp[:, selected_features] = 1
                explanation.append(exp)
                anomalies_info[anomaly] = {
                "alpha": alpha,
                "features": selected_features
                }
            elif anomaly == "local":
                selected_features = anomalies_info["features"]
                alpha = anomalies_info["alpha"]
                gm_copy = copy.deepcopy(graphical_model)
                covariances = gm_copy.covariances_
                covariances[:, selected_features, selected_features] = (
                    alpha * covariances[:, selected_features, selected_features]
                )
                gm_copy.covariances_ = covariances
                # Generate the anomalies
                anomalies = gm_copy.sample(anomalies_nb[anomaly])[0]
                X_anomalies.append(anomalies)
                y_anomalies.append(np.full(anomalies_nb[anomaly], anomaly_code[anomaly]))
                exp = np.zeros(anomalies.shape)
                exp[:, selected_features] = 1
                explanation.append(exp)
                anomalies_info[anomaly] = {
                    "alpha": alpha,
                    "features": selected_features,
                }
            elif anomaly == "additive_noise":
                selected_features = anomalies_info["features"]
                alpha = anomalies_info["alpha"]
                anomalies = graphical_model.sample(anomalies_nb[anomaly])[0]
                noise = np.random.normal(
                    anomalies[:, selected_features].std() * alpha,
                    anomalies[:, selected_features].std(),
                    anomalies[:, selected_features].shape,
                )
                anomalies[:, selected_features] += noise
                X_anomalies.append(anomalies)
                y_anomalies.append(np.full(anomalies_nb[anomaly], anomaly_code[anomaly]))
                exp = np.zeros(anomalies.shape)
                exp[:, selected_features] = 1
                explanation.append(exp)
                anomalies_info[anomaly] = {
                    "features": selected_features,
                    "alpha": alpha,
                }
            elif anomaly == "multiplicative_noise":
                selected_features = anomalies_info["features"]
                alpha = anomalies_info["alpha"]
                anomalies = graphical_model.sample(anomalies_nb[anomaly])[0]

                noise = np.random.normal(
                    anomalies[:, selected_features].std() * alpha,
                    0.3,
                    anomalies[:, selected_features].shape,
                )
                anomalies[:, selected_features] *= noise
                X_anomalies.append(anomalies)
                y_anomalies.append(np.full(anomalies_nb[anomaly], anomaly_code[anomaly]))
                exp = np.zeros(anomalies.shape)
                exp[:, selected_features] = 1
                explanation.append(exp)
                anomalies_info[anomaly] = {
                    "features": selected_features,
                    "alpha": alpha,
                }
            else:
                raise ValueError("Anomaly type not recognized")
        return X_anomalies, y_anomalies, explanation, anomalies_info

    def add_duplicated_anomalies(self, X, y, duplicate_times: int):
        if duplicate_times <= 1:
            pass
        else:
            # index of normal and anomaly data
            idx_n = np.where(y == 0)[0]
            idx_a = np.where(y == 1)[0]

            # generate duplicated anomalies
            idx_a = np.random.choice(idx_a, int(len(idx_a) * duplicate_times))

            idx = np.append(idx_n, idx_a)
            random.shuffle(idx)
            X = X[idx]
            y = y[idx]

        return X, y

    def generate_realistic_synthetic(
        self, X, y, anomalies: list
    ):
        """
        Currently, four types of realistic synthetic outliers can be generated:
        1. local outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM
        distribution with modified covariance
        2. global outliers: where normal data follows the GMM distribuion, and anomalies follow the uniform distribution
        3. dependency outliers: where normal data follows the vine coupula distribution, and anomalies
        follow the independent distribution captured by GaussianKDE
        4. cluster outliers: where normal data follows the GMM distribuion, and anomalies follow the
        GMM distribution with modified mean

        :param X: input X
        :param y: input y
        :param realistic_synthetic_mode: the type of generated outliers
        :param alpha: the scaling parameter for controling the generated local and cluster anomalies
        :param percentage: controling the generated global anomalies
        """
        # the number of normal data and anomalies
        pts_n = len(np.where(y == 0)[0])
        pts_a = len(np.where(y == 1)[0])

        nb_anomalies = [int(pts_a / len(anomalies))] * len(anomalies)
        # only use the normal data to fit the model
        X = X[y == 0]
        y = y[y == 0]

        anomaly_scheme = generate_anomalies_scheme(anomalies, X.shape[1])
        metric_list = []
        n_components_list = list(np.arange(1, 10))

        for n_components in n_components_list:
            gm = GaussianMixture(
                n_components=n_components, random_state=self.seed
            ).fit(X)
            metric_list.append(gm.bic(X))

        best_n_components = n_components_list[np.argmin(metric_list)]

        # refit based on the best n_components
        gm = GaussianMixture(
            n_components=best_n_components, random_state=self.seed
        ).fit(X)

        # generate the synthetic normal data
        X_synthetic_normal = gm.sample(pts_n)[0]

        # Anomaly generation
        anomalies_nb = {k: nb_anomalies[i] for i, k in enumerate(anomalies)}
        X_synthetic_anomalies, y_anomalies, explanation_anomalies, anomalies_info = \
            self.add_anomalies(X_normal=X_synthetic_normal,
                               graphical_model=gm,
                               anomalies_scheme=anomaly_scheme,
                               anomalies_nb=anomalies_nb
        )

        X = np.concatenate((X_synthetic_normal, *X_synthetic_anomalies), axis=0)
        y = np.append(
            np.repeat(0, X_synthetic_normal.shape[0]),
            y_anomalies,
        )
        explanation = np.concatenate((np.zeros(X_synthetic_normal.shape), *explanation_anomalies), axis=0)

        return X, y, explanation, anomalies_info

    def generator(
        self,
        X=None,
        y=None,
        scale=True,
        la=None,
        at_least_one_labeled=False,
        noise_type=None,
        duplicate_times: int = 2,
        max_size=10000,
        synthetic_anomalies=None,
        alpha: int = 5,
        percentage=0.1,
    ):
        """
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        """

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is None:
            assert (
                X is not None and y is not None
            ), "For customized dataset, you should provide the X and y!"
        else:
            if self.dataset in self.dataset_list_classical:
                data = np.load(
                    os.path.join(DATA_PATH, "Classical", self.dataset + ".npz"),
                    allow_pickle=True,
                )
            elif self.dataset in self.dataset_list_cv:
                data = np.load(
                    os.path.join(
                        DATA_PATH, "CV_by_ResNet18", self.dataset + ".npz"
                    ),
                    allow_pickle=True,
                )
            elif self.dataset in self.dataset_list_nlp:
                data = np.load(
                    os.path.join(
                        DATA_PATH, "NLP_by_BERT", self.dataset + ".npz"
                    ),
                    allow_pickle=True,
                )
            elif "synthetic" in self.dataset:
                data = np.load(
                    self.config.dataset.dataset_path, allow_pickle=True
                )
            else:
                raise NotImplementedError(f"Dataset {self.dataset} is not supported!")
            X = data["X"]
            y = data["y"]
            if "explanation" in data.keys():
                explanation = data["explanation"]
            else:
                explanation = np.zeros_like(X)

        # if the dataset is too small, generating duplicate smaples up to n_samples_threshold
        if len(y) < self.n_samples_threshold and self.generate_duplicates:
            self.logger.info(f"generating duplicate samples for dataset {self.dataset}...")
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(
                np.arange(len(y)), self.n_samples_threshold, replace=True
            )
            X = X[idx_duplicate]
            y = y[idx_duplicate]
            explanation = explanation[idx_duplicate]

        # if the dataset is too large, subsampling for considering the computational cost
        if len(y) > max_size:
            self.logger.info(f"subsampling for dataset {self.dataset}...")
            self.utils.set_seed(self.seed)
            idx_sample = np.random.choice(
                np.arange(len(y)), max_size, replace=False
            )
            X = X[idx_sample]
            y = y[idx_sample]
            explanation = explanation[idx_sample]

        anomalies_info = None
        if synthetic_anomalies is not None:
            # we save the generated dependency anomalies, since the Vine Copula could spend too long for generation
            X, y, explanation, anomalies_info = self.generate_realistic_synthetic(
                X,
                y,
                anomalies=synthetic_anomalies,
            )

        # show the statistic
        self.utils.data_description(X=X, y=y)

        # spliting the current data to the training set and testing set
        if not self.normal:
            if self.test_size == 0:
                X_train = X.copy()
                y_train = y.copy()
                X_test = X.copy()
                y_test = y.copy()
                explanation_train = explanation.copy()
                explanation_test = explanation.copy()
            else:
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    explanation_train,
                    epxlanation_test,
                ) = train_test_split(
                    X,
                    y,
                    explanation,
                    test_size=self.test_size,
                    shuffle=True,
                    stratify=y,
                )

        else:
            indices = np.arange(len(X))
            normal_indices = indices[y == 0]
            anomaly_indices = indices[y > 0]

            train_size = round((1 - self.test_size) * normal_indices.size)
            train_indices, test_indices = (
                normal_indices[:train_size],
                normal_indices[train_size:],
            )
            test_indices = np.append(test_indices, anomaly_indices)

            X_train = X[train_indices]
            y_train = y[train_indices]
            explanation_train = explanation[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
            explanation_test = explanation[test_indices]

        # standard scaling
        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            col_mean = np.nanmean(X_train, axis=0)
            inds = np.where(np.isnan(col_mean))
            col_mean[inds] = 0

            inds = np.where(np.isnan(X_train))
            X_train[inds] = np.take(col_mean, inds[1])

            col_mean = np.nanmean(X_test, axis=0)
            inds = np.where(np.isnan(col_mean))
            col_mean[inds] = 0

            inds = np.where(np.isnan(X_test))
            X_test[inds] = np.take(col_mean, inds[1])

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if type(la) is float:
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(
                    idx_anomaly, ceil(la * len(idx_anomaly)), replace=False
                )
            else:
                idx_labeled_anomaly = np.random.choice(
                    idx_anomaly, int(la * len(idx_anomaly)), replace=False
                )
        elif type(la) is int:
            if la > len(idx_anomaly):
                raise AssertionError(
                    f"the number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !"
                )
            else:
                idx_labeled_anomaly = np.random.choice(
                    idx_anomaly, la, replace=False
                )
        else:
            raise NotImplementedError

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)

        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        return {
            "X": X,
            "y": y,
            "explanation": explanation,
        }, anomalies_info


def generate_and_save_synthethic_data(cfg, saving_path, db_name):
    if cfg.training_method.name == "semi-supervised":
        datagenerator = DataGenerator(
            seed=cfg.random_seed, test_size=0.5, normal=True, config=cfg
        )  # data generator
    else:
        datagenerator = DataGenerator(
            seed=cfg.random_seed, test_size=0, normal=False, config=cfg
        )  # data generator

    datagenerator.dataset = cfg.dataset.dataset_name  # specify the dataset name
    data, anomaly_information = datagenerator.generator(
        la=0,
        max_size=50000,
        synthetic_anomalies=cfg.realistic_synthetic_mode,
        alpha=cfg.alpha,
        percentage=cfg.percentage,
    )  # maximum of 50,000 data points are available
    X = data['X']
    y = data['y']
    explanation = ['explanation']
    print(saving_path, db_name)
    with open(os.path.join(saving_path, "anomaly_information.yaml"), "w") as file:
        yaml.dump(anomaly_information, file)
    pickle.dump(data, open(os.path.join(saving_path, "data.npy"), "wb"))
    np.save(os.path.join(saving_path, "features.npy"), np.array(X))
    np.save(os.path.join(saving_path, "labels.npy"), np.array(y))
    np.save(
        os.path.join(saving_path, "explanation.npy"), np.array(explanation)
    )
    # Generate yalm config file:
    config = {
        "data_type": cfg.dataset.data_type,
        "dataset_name": db_name,
        "dataset_path": os.path.join(saving_path, "data.npy"),
        "test_ratio": 0.2,
    }
    config_path = "conf/dataset"
    with open(os.path.join(config_path, f"{db_name}.yaml"), "w") as file:
        yaml.dump(config, file)

@hydra.main(version_base=None, config_path="../../conf", config_name="config_datagen")
def main(cfg: omegaconf.DictConfig):
    db_name = (
        f"ADBench_synthetic_{cfg.dataset.dataset_name}"
    )
    saving_path = cfg.output_path + "/ADBench_synthetic/"+ db_name
    os.makedirs(saving_path, exist_ok=True)
    generate_and_save_synthethic_data(cfg, saving_path, db_name)


if __name__ == "__main__":
    main()
