import logging
import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

from utils import pred_from_scores


class WeightedLossIterativeLearning:
    def __init__(
        self, max_round, epochs, batch_size, lr, device, model, model_config
    ):
        self.max_round = max_round
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config

    def __call__():
        pass

    def termination_critertion(self, scores_t, r_t_minus_one):
        """
        Termination criterion defined in https://arxiv.org/pdf/2309.09436
        """
        sorted_indices = np.argsort(scores_t)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(scores_t))
        r_t = np.argsort(scores_t)

        threshold = np.ceil(np.array(len(r_t) // 2))

        cond1 = (r_t < threshold) & (r_t_minus_one > threshold)
        cond2 = (r_t > threshold) & (r_t_minus_one < threshold)
        A = cond1 | cond2
        h = np.sum(A)
        return h, r_t

    def update_weights(self, scores_t, tau=1):
        scores_t_median = np.median(scores_t)
        alpha_t = 1 / (
            min(
                scores_t_median - np.min(scores_t),
                scores_t.max() - scores_t_median,
            )
            * tau
        )
        beta_t = scores_t_median
        weights = 1 / (1 + np.exp(alpha_t * (scores_t - beta_t)))
        return weights

    def train(self, X_train, y_train=None, X_eval=None, y_eval=None):
        weights = np.ones(len(y_train))
        best_model = None
        best_h = np.inf
        r_t = np.arange(len(y_train))
        train_log = {
            "train_losses": [],
            "f1_scores": [],
            "roc_auc_scores": [],
        }
        for iteration in range(self.max_round):
            model = deepcopy(self.model)
            model, train_losses = model.fit(
                X_train, weights=weights, model_config=self.model_config
            )
            train_log["train_losses"].append(train_losses[-1])
            anomaly_scores = model.predict_score(X_train)
            h, r_t = self.termination_critertion(
                scores_t=anomaly_scores, r_t_minus_one=r_t
            )
            weights = self.update_weights(anomaly_scores)

            if h < best_h:
                best_model = model
                best_h = h
            if X_eval is not None and y_eval is not None:
                scores = model.predict_score(X_eval)
                nb_anomalies = np.sum(y_eval)
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
        return best_model, train_log

    def plot_training_log(self, train_log):

        plt.figure()

        plt.plot(train_log["train_losses"], label="train loss")
        plt.plot(train_log["f1_scores"], label="f1 score")
        plt.plot(train_log["roc_auc_scores"], label="roc auc score")
        plt.plot(train_log["nb_anomalies"], label="nb anomalies")

        plt.legend()
        plt.xlabel("iteration")

        plt.title("Training log")
        plt.grid()

        plt.xticks(range(len(train_log["train_losses"])))
        plt.savefig(os.path.join(self.saving_path, "train_log.png"))
