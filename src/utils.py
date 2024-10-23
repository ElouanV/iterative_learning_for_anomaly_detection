import os
from pathlib import Path

import numpy as np

from dataset.data_generator import DataGenerator
from models.ddpm.ddpm import DDPM
from models.deepsvdd.deepSVDD import DeepSVDD
from models.dte import DTECategorical, DTEInverseGamma


def select_model(model_config: dict, device):
    if model_config.model_name == "DTECategorical":
        return DTECategorical(
            hidden_size=model_config.model_parameters.hidden_size,
            epochs=model_config.training.epochs,
            batch_size=model_config.training.batch_size,
            lr=model_config.training.learning_rate,
            num_bins=model_config.model_parameters.num_bins,
            device=device,
        )
    elif model_config.model_name == "DTEInverseGamma":
        return DTEInverseGamma(
            hidden_size=model_config.model_parameters.hidden_size,
            epochs=model_config.training.epochs,
            batch_size=model_config.training.batch_size,
            lr=model_config.training.learning_rate,
            device=device,
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
            device=device,
            T=model_config.model_parameters.T,
        )
    elif model_config.model_name == "DeepSVDD":
        model = DeepSVDD(
            nu=model_config.model_parameters.nu,
        )
        model.set_network(
            model_config.model_parameters.net_name,
            input_dim=model_config.model_parameters.input_dim,
            model_config=model_config,
        )
        return model


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


def get_dataset(cfg: Path):
    if cfg.training_method.name == "semi-supervised":
        datagenerator = DataGenerator(
            seed=cfg.random_seed, test_size=0.5, normal=True, config=cfg
        )  # data generator
    else:
        datagenerator = DataGenerator(
            seed=cfg.random_seed, test_size=0, normal=False, config=cfg
        )  # data generator

    datagenerator.dataset = cfg.dataset.dataset_name  # specify the dataset name
    data = datagenerator.generator(
        la=0, max_size=50000
    )  # maximum of 50,000 data points are available
    if cfg.training_method.name != "semi-supervised":
        X = data["X_test"]
        y = data["y_test"]

        indices = np.arange(len(X))
        subset = np.random.choice(indices, size=len(indices), replace=True)

        data = {}
        data["X_train"] = X[subset]
        data["y_train"] = y[subset]

        data["X_test"] = X
        data["y_test"] = y
    return data


def setup_experiment(cfg: dict):
    experiment_name = (
        f"{cfg.model.model_name}_{cfg.training_method.name}_{cfg.training_method.sampling_method}"
        + (
            f"_{cfg.training_method.ratio}"
            if cfg.training_method.name == "dataset_sampling"
            else ""
        )
        + f"_s{cfg.random_seed}"
    )
    saving_path = Path(
        cfg.output_path,
        str(cfg.run_id),
        experiment_name,
        cfg.dataset.dataset_name,
    )
    os.makedirs(saving_path, exist_ok=True)
    return saving_path


def check_cuda(logger, device=None):
    import torch

    if torch.cuda.is_available():
        print(f"Cuda version: {torch.version.cuda}")
    else:
        logger.warning("Cuda not available")
    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"Device: {device}")
    return device


def low_density_anomalies(test_log_probs, num_anomalies):
    """Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob"""
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies - 1)[
        :num_anomalies
    ]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds
