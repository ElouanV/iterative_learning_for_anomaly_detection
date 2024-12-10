import logging
import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from utils import pred_from_scores, select_model
from viz.training_viz import iterative_training_score_evolution


class SamplingIterativeLearning:
    def __init__(self, conf: dict, exp_name: str, saving_path: Path = None):
        self.model_config = conf.model
        self.conf = conf
        self.update_trainset = self.get_sampling_method(conf)
        self.logger = logging.getLogger(__name__)
        self.saving_path = saving_path
        self.exp_name = exp_name

    @staticmethod
    def get_sampling_method(conf: dict):
        conf = conf.training_method
        if conf.ratio == "cosine":
            return CosineSampling(conf.nu_min, conf.nu_max, conf.max_iter)
        elif conf.ratio == "exponential":
            return ExponentialSampling(conf.nu_min, conf.nu_max, conf.max_iter)
        elif type(conf.ratio) is float:
            return ConstantSampling(conf.ratio)
        else:
            raise ValueError(
                f"Unknown ratio method: {conf.ratio}, type: {type(conf.ratio)}"
            )

    def train(
        self,
        X_train,
        y_train=None,
        X_eval=None,
        y_eval=None,
        model=None,
        max_iter=10,
        device="cuda",
    ):

        if model is None:
            model = select_model(self.config.model)

        current_X = X_train.copy()
        current_y = y_train.copy()
        train_log = {
            "train_losses": [],
            "f1_scores": [],
            "roc_auc_scores": [],
            "nb_anomalies": [],
        }
        iteration_scores = []
        for iteration in range(max_iter):
            train_log["nb_anomalies"].append(
                np.sum(current_y) / np.sum(y_train)
            )
            current_model = deepcopy(model)

            model, train_loss = current_model.fit(current_X)
            scores = current_model.predict_score(X_train)
            current_X, current_y = self.update_trainset(
                scores, X_train, y_train, iteration
            )
            train_log["train_losses"].append(train_loss[-1])
            # Save scores
            np.save(
                os.path.join(self.saving_path, f"scores_{iteration}.npy"),
                scores,
            )

            iteration_scores.append(scores)
            # Evaluation
            if X_eval is not None and y_eval is not None:
                scores = current_model.predict_score(X_eval)
                nb_anomalies = int(np.sum(y_eval))
                y_pred = pred_from_scores(scores, nb_anomalies)
                f1 = f1_score(y_eval, y_pred)
                self.logger.info(f"F1 score at iteration {iteration}: {f1}")
                roc = roc_auc_score(y_eval, y_pred)
                self.logger.info(
                    f"ROC AUC score at iteration {iteration}: {roc}"
                )
                train_log["f1_scores"].append(f1)
                train_log["roc_auc_scores"].append(roc)
        self.plot_training_log(train_log)
        iterative_training_score_evolution(
            iteration_scores,
            exp_name=self.exp_name,
            saving_path=self.saving_path,
        )
        return current_model, train_log

    def plot_training_log(self, train_log):

        plt.figure()

        plt.plot(train_log["train_losses"], label="train loss")
        plt.plot(train_log["f1_scores"], label="f1 score")
        plt.plot(train_log["roc_auc_scores"], label="roc auc score")
        plt.plot(train_log["nb_anomalies"], label="nb anomalies")

        plt.legend()
        plt.xlabel("iteration")
        plt.grid()

        plt.title("Training log")

        plt.xticks(range(len(train_log["train_losses"])))
        plt.savefig(os.path.join(self.saving_path, "train_log.png"))


class SamplingMethod:
    def __init__(self, method="deterministic"):
        self.method = method
        pass

    def __call__(self, scores, X, y, iteration_number) -> tuple:
        pass

    def get_current_ratio(self, iteration_number):
        pass


class ConstantSampling(SamplingMethod):
    def __init__(self, ratio: float, method="deterministic"):
        super().__init__(method)
        self.ratio = ratio

    def __call__(self, scores, X, y, iteration_number=None):
        """
        Select the ratio% lowest scores
        """
        if self.method == "deterministic":
            indices_sorted = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=False
            )
            indices_to_keep = indices_sorted[: int(len(scores) * self.ratio)]
            return X[indices_to_keep], y[indices_to_keep]
        elif self.method == "probabilistic":
            # Sampling with probability proportional to the score
            proba = scores / np.sum(scores)
            indices = np.random.choice(
                range(len(scores)), size=int(len(scores) * self.ratio), p=proba
            )
            return X[indices], y[indices]

    def get_current_ratio(self, iteration_number):
        return self.ratio


class CosineSampling(SamplingMethod):
    def __init__(
        self, nu_min: float, nu_max: float, max_iter=10, method="deterministic"
    ):
        super().__init__(method)
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.max_iter = max_iter

    def __call__(self, scores, X, y, iteration_number):
        """
        Select the ratio% lowest scores
        """
        ratio = self.nu_min + 1 / 2 * (self.nu_max - self.nu_min) * (
            1 + np.cos(np.pi * iteration_number / self.max_iter)
        )
        if self.method == "deterministic":
            indices_sorted = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=False
            )
            indices = indices_sorted[: int(len(scores) * ratio)]
            return X[indices], y[indices]
        elif self.method == "probabilistic":
            # Sampling with probability proportional to the score
            proba = scores / np.sum(scores)
            indices = np.random.choice(
                range(len(scores)), size=int(len(scores) * ratio), p=proba
            )
            return X[indices], y[indices]

    def get_current_ratio(self, iteration_number):
        return self.nu_min + 1 / 2 * (self.nu_max - self.nu_min) * (
            1 + np.cos(np.pi * iteration_number / self.max_iter)
        )


class ExponentialSampling(SamplingMethod):
    def __init__(
        self, nu_min: float, nu_max: float, max_iter=10, method="deterministic"
    ):
        super().__init__(method)
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.max_iter = max_iter

    def __call__(self, scores, X, y, iteration_number) -> tuple:
        """
        Select the ratio% lowest scores
        """
        ratio = self.nu_min + 1 / 2 * (self.nu_max - self.nu_min) * (
            1 + np.exp(-iteration_number / self.max_iter)
        )
        if self.method == "deterministic":
            indices_sorted = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=False
            )
            indices = indices_sorted[: int(len(scores) * ratio)]
            return X[indices], y[indices]
        elif self.method == "probabilistic":
            # Sampling with probability proportional to the score
            proba = scores / np.sum(scores)
            indices = np.random.choice(
                range(len(scores)), size=int(len(scores) * ratio), p=proba
            )
            return X[indices], y[indices]

    def get_current_ratio(self, iteration_number):
        return self.nu_min + 1 / 2 * (self.nu_max - self.nu_min) * (
            1 + np.exp(-iteration_number / self.max_iter)
        )
