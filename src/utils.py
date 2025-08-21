import os
from pathlib import Path

import numpy as np
import torch

from dataset.data_generator import DataGenerator
from models.ddpm import DDPM
from models.dte import DTECategorical, DTEInverseGamma


def select_model(model_config: dict, device):
    if model_config.model_name == "DTEC":
        return DTECategorical(
            hidden_size=model_config.model_parameters.hidden_size,
            epochs=model_config.training.epochs,
            batch_size=model_config.training.batch_size,
            lr=model_config.training.learning_rate,
            num_bins=model_config.model_parameters.num_bins,
            T=model_config.model_parameters.T,
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


def pred_from_scores(scores, num_anomalies: int):
    """Return binary prediction vector with 1 for the top num_anomalies scores."""
    indices_sorted = np.argsort(scores)[::-1]
    result = np.zeros(len(scores))
    result[indices_sorted[:num_anomalies]] = 1
    return result


def get_dataset(cfg):
    """Load dataset via DataGenerator based on the training method (semi-supervised vs unsupervised)."""
    if cfg.training_method.name == "semi-supervised":
        datagenerator = DataGenerator(
            seed=cfg.random_seed, test_size=0.5, normal=True, config=cfg
        )
    else:
        datagenerator = DataGenerator(
            seed=cfg.random_seed, test_size=0, normal=False, config=cfg
        )
    datagenerator.dataset = cfg.dataset.dataset_name
    data = datagenerator.generator(
        la=0,
        max_size=50000,
        alpha=cfg.alpha,
        percentage=cfg.percentage,
    )
    return data


def setup_experiment(cfg: dict):
    """
    Sets up the experiment by generating the experiment name and creating the necessary directories.
    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - model (object): An object with attributes:
                - model_name (str): Name of the model.
                - model_parameters (object): An object with attributes:
                    - T (int): Temperature parameter for the model (if applicable).
                    - num_bins (int): Number of bins for the model (if applicable).
            - training_method (object): An object with attributes:
                - name (str): Name of the training method.
                - sampling_method (str): Sampling method used in training.
                - ratio (float): Ratio parameter for the training method (if applicable).
            - random_seed (int): Random seed for reproducibility.
            - output_path (str): Base path for saving experiment outputs.
            - run_id (int): Unique identifier for the current run.
            - dataset (object): An object with attributes:
                - dataset_name (str): Name of the dataset.
            - realistic_synthetic_mode (list): List of modes for realistic synthetic data (if applicable).
    Returns:
        tuple: A tuple containing:
            - saving_path (Path): The path where experiment outputs will be saved.
            - experiment_name (str): The generated name of the experiment.
    """
    experiment_name = (
        f"{cfg.model.model_name}_{cfg.training_method.name}_{cfg.training_method.sampling_method}"
        + (
            f"_{cfg.training_method.ratio}"
            if cfg.training_method.name == "DSIL"
            else ""
        )
        + f"_s{cfg.random_seed}"
        + (
            f"_T{cfg.model.model_parameters.T}_bins{cfg.model.model_parameters.num_bins}"
            if cfg.model.model_name == "DTEC"
            else ""
        )
    )

    if cfg.training_method.name == "DSIL":
        try:
            if cfg.training_method.epoch_budget:
                experiment_name += "_epoch_budget"
        except AttributeError:
            pass
        try:
            if cfg.training_method.reinitialize_model_weights:
                experiment_name += "_reinitialize_model_weights"
            else:
                experiment_name += "_no_reinitialize_model_weights"
        except AttributeError:
            pass

    saving_path = Path(
        cfg.output_path,
        str(cfg.run_id),
        experiment_name,
        (
            cfg.dataset.dataset_name
            + (
                f"_{'_'.join(cfg.realistic_synthetic_mode)}"
                if cfg.realistic_synthetic_mode
                else ""
            )
        ),
    )
    os.makedirs(saving_path, exist_ok=True)
    return saving_path, experiment_name


def check_cuda(logger, device=None):
    """Return torch.device respecting explicit user setting or auto-detecting CUDA."""
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


def dataframe_to_latex(
    df,
    column_format=None,
    caption=None,
    label=None,
    header=True,
    index=False,
    float_format="%.2f",
):
    """
    Convert a pandas DataFrame to a clean LaTeX table.

    Parameters:
        df (pd.DataFrame): The DataFrame to convert.
        column_format (str): Column alignment (e.g., 'lcr' for left, center, right).
                            Defaults to auto-detection based on the number of columns.
        caption (str): Caption for the table.
        label (str): Label for referencing the table in LaTeX.
        header (bool): Whether to include column headers.
        index (bool): Whether to include the DataFrame index.
        float_format (str): Format for floating-point numbers (default: "%.2f").

    Returns:
        str: The LaTeX table as a string.
    """
    # Default column format if not specified
    if column_format is None:
        column_format = "l" * (len(df.columns) + (1 if index else 0))

    # Convert the DataFrame to LaTeX
    latex_str = df.to_latex(
        index=index,
        column_format=column_format,
        header=header,
        float_format=float_format,
        escape=False,  # Allows LaTeX-specific characters (e.g., \%)
    )

    # Add optional caption and label
    if caption or label:
        latex_table = "\\begin{table}[ht]\n\\centering\n"
        if caption:
            latex_table += f"\\caption{{{caption}}}\n"
        if label:
            latex_table += f"\\label{{{label}}}\n"
        latex_table += latex_str
        latex_table += "\\end{table}"
    else:
        latex_table = latex_str

    return latex_table
