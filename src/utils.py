import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path

from models.dte import DTECategorical, DTEInverseGamma
from src.models.ddpm import DDPM


def configure_logger(saving_path: Path):
    """
    Configures the logger for the application with file and console handlers
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s] - [%(levelname)s] - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler and set level to debug
    file_handler = logging.FileHandler(os.path.join(saving_path, "run.log"))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] - [%(levelname)s] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def plot_f1_and_loss(
    data, dataset_name, X, num_anomalies, percentage_keep, num_iteration
):

    # Create a figure with 2 subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # First subplot: F1 scores
    axs[0].plot(data["F1_scores"], label="F1_scores")
    axs[0].plot(
        data["nb_anos_in_training"], label="nb_anos_in_training", color="green"
    )
    axs[0].set_xlabel("itération")
    axs[0].set_ylabel("score")
    axs[0].set_title("Evolution scores")
    axs[0].grid()
    axs[0].legend()

    # Second subplot: Training Loss
    n_point_loss = num_iteration + 1
    indices_plot = np.linspace(
        0, len(data["train_losses"]) - 1, n_point_loss
    ).astype(int)
    train_losses_sampled = [data["train_losses"][i] for i in indices_plot]
    axs[1].plot(
        np.linspace(0, len(data["F1_scores"]) - 1, n_point_loss),
        train_losses_sampled,
        label="Train Loss",
    )
    axs[1].set_xlabel("itération")
    axs[1].set_title("Evolution Loss")
    axs[1].grid()
    axs[1].legend()

    fig.suptitle(
        f"{dataset_name}\nNombre d'observations: {X.shape[0]}\n Nombre de features: {X.shape[1]}\n \
            proportion d'anomalies: {(num_anomalies*100/X.shape[0]):.2f}%",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, bottom=0.2)
    plt.text(
        0.5,
        0.01,
        f"Métriques additionnelles:\n E(F1) =  {np.mean(data['F1_scores']):.2f}\n",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
        fontsize=12,
    )

    plt.savefig(f"./res/{dataset_name}/summary_p_{str(percentage_keep)}.png")
    plt.close()


def select_model(model_config: dict):
    if model_config.model_name == "DTECategorical":
        return DTECategorical(
            hidden_size=model_config.model_parameters.hidden_size,
            epochs=model_config.training.epochs,
            batch_size=model_config.training.batch_size,
            lr=model_config.training.learning_rate,
            num_bins=model_config.model_parameters.num_bins,
        )
    elif model_config.model_name == "DTEInverseGamma":
        return DTEInverseGamma(
            hidden_size=model_config.model_parameters.hidden_size,
            epochs=model_config.training.epochs,
            batch_size=model_config.training.batch_size,
            lr=model_config.training.learning_rate,
        )
    elif model_config.model_name == "DDPM":
        resnet_params = {
            "d_main": model_config.model_parameters.d_main,
            "n_blocks": model_config.model_parameters.n_blocks,
            "d_hidden": model_config.model_parameters.d_hidden,
            "dropout_first": model_config.model_parameters.dropout_first,
            "dropout_second": model_config.model_parameters.dropout_second,
        }
        return DDPM(
            epochs=model_config.training.epochs,
            batch_size=model_config.training.batch_size,
            lr=model_config.training.learning_rate,
            resnet_parameters=resnet_params,
        )


def count_ano(indices, y):
    """
    count the number of anomalies among indices
    """
    return sum(y[indices])


def pred_from_scores(scores, num_anomalies):
    """
    retourne une prédiction y où
    y[i] = 1 si scores[i] est l'un des num_anomalies plus grand score
        et 0 sinon.
    """

    indices_sorted = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )
    indices_sorted = np.argsort(scores)[::-1]
    result = np.zeros(len(scores))
    result[indices_sorted[:num_anomalies]] = 1
    return result


def get_normal_indices(
    scores,
    p,
    method="constant",
    iteration=-1,
    nu_min=50,
    nu_max=100,
    max_iter=10,
):
    """
    takes :
    |   scores : t_pred obtained by DTE model
    |   p : percentage of indices that we want to keep for training
    return :
    |   output : list of indices that are the most likely to be normal and should be use for next training phase
    """
    if method == "cosine":
        p = nu_min + 1 / 2 * (nu_max - nu_min) * (
            1 + np.cos(np.pi * iteration / max_iter)
        )
    n = scores.shape[0]
    indices_sorted = sorted(range(len(scores)), key=lambda i: scores[i])
    return indices_sorted[: int(n * p)]


def get_dataset(data_path: Path):
    extension = data_path.suffix
    X, y = None, None
    if extension == ".csv":
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    elif extension == ".npz":
        data = np.load(data_path, allow_pickle=True)
        X, y = data["X"], data["y"]
    return X, y
