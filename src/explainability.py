import logging
import time
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
from adbench.myutils import Utils

from metrics import explanation_accuracy, nDCG_p
from shap_explainer import ShapExplainer
from utils import (check_cuda, get_dataset, low_density_anomalies,
                   select_model, setup_experiment)


def explanation(
    model, method, data, samples, expected_explanation, saving_path, exp_name
):
    start_time = time.time()
    if method == "SHAP":
        shap_explainer = ShapExplainer(model, data["X_train"])
        explanation = shap_explainer.explain_instance(
            samples,
        ).squeeze()
    elif method == "grad":
        explanation = model.gradient_explanation(
            samples,
        )
    elif method == "mean_diffusion_perturbation":
        explanation = model.instance_explanation(
            samples, saving_path=saving_path, step=10, agg="mean"
        )
    elif method == "max_diffusion_perturbation":
        explanation = model.instance_explanation(
            samples, saving_path=saving_path, step=10, agg="max"
        )
    elif method == "reconstruction_error" and model.model_name == "DDPM":
        explanation = model.instance_explanation(
            samples,
            saving_path=saving_path,
            step=10,
            exp_name=exp_name,
            expected_explanation=expected_explanation,
        )
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    end_time = time.time()
    explanation_time = end_time - start_time
    nDCG = nDCG_p(
        importance_scores=explanation, relevance_matrix=expected_explanation
    )
    accuracy = explanation_accuracy(
        explanation=explanation, ground_truth=expected_explanation
    )
    # Save explanation and expected explanation
    np.save(Path(saving_path, f"{method}_explanation.npy"), explanation)
    np.save(Path(saving_path, f"{method}_expected_explanation.npy"), expected_explanation)
    return nDCG.mean(), accuracy.mean(), explanation_time, explanation


def run_config(cfg, logger, device):
    utils = Utils()  # utils function
    utils.set_seed(cfg.random_seed)
    saving_path, experiment_name = setup_experiment(cfg)
    data = get_dataset(cfg)
    logger.info(
        f"Model name: {cfg.model.model_name}, dataset name: {cfg.dataset.dataset_name}, training method:"
        f" {cfg.training_method.name}, sampling name(if applicable):"
        f" {cfg.training_method.sampling_method}, ratio(for iterative learning"
        f" sampling method only): "
        f"{cfg.training_method.ratio if cfg.training_method =='iterative_dataset_sampling' else None}, random seed:   \
            {cfg.random_seed}"
    )

    model = select_model(cfg.model, device=device)
    model_weights_path = Path(saving_path, "model.pth")
    if model_weights_path.exists():
        model.load_model(model_weights_path, X=data["X_train"])
    else:
        raise FileNotFoundError(
            f"Model weights not found in {model_weights_path}. If you want to train the model with the given"
            "configuration, use the 'train' script and run this script again."
        )

    score = model.predict_score(data["X_test"], device=device).squeeze()
    data["y_test"][data["y_test"] > 0] = 1
    metric_df = pd.read_csv(Path(saving_path, "model_metrics.csv"))
    force_rerun = cfg.force_rerun
    indices = np.arange(len(data["y_test"]))
    # Suppose we know how much anomaly are in the dataset
    y_pred = low_density_anomalies(-score, len(indices[data["y_test"] == 1]))
    samples_indices = np.arange(len(data["X_test"]))[y_pred == 1]
    samples = data["X_test"][samples_indices]
    expected_explanation = data["explanation_test"][samples_indices]
    existing_columns = metric_df.columns
    np.save(Path(saving_path, "samples_to_explain.npy"), samples)
    samples_labels = data["y_test"][samples_indices]
    np.save(Path(saving_path, "samples_to_explain_labels.npy"),  samples_labels)
    if cfg.model.model_name == "DTEC":
        if "mean_diffusion_accuracy" not in existing_columns or force_rerun:
            (
                mean_diffusion_nDCG,
                mean_diffusion_accuracy,
                mean_diffusion_explanation_time,
                mean_diffusion_explanation,
            ) = explanation(
                model,
                "mean_diffusion_perturbation",
                data,
                samples,
                expected_explanation,
                saving_path,
                experiment_name,
            )
            logger.info(
                f"Mean diffusion feat importance NDCG: {mean_diffusion_nDCG.mean()}"
            )
            metric_df["mean_diffusion_accuracy"] = mean_diffusion_accuracy
            metric_df["mean_diffusion_ndcg"] = mean_diffusion_nDCG
            metric_df["mean_diffusion_time"] = mean_diffusion_explanation_time
        if "max_diffusion_accuracy" not in existing_columns or force_rerun:
            (
                max_diffusion_nDCG,
                max_diffusion_accuracy,
                max_diffusion_explanation_time,
                max_diffusion_explanation,
            ) = explanation(
                model,
                "max_diffusion_perturbation",
                data,
                samples,
                expected_explanation,
                saving_path,
                experiment_name,
            )
            logger.info(
                f"Max diffusion feature importance NDCG: {max_diffusion_nDCG.mean()}"
            )
            metric_df["max_diffusion_accuracy"] = max_diffusion_accuracy
            metric_df["max_diffusion_ndcg"] = max_diffusion_nDCG
            metric_df["max_diffusion_time"] = max_diffusion_explanation_time
        if "shap_explanation_accuracy" not in existing_columns or force_rerun:
            (
                shap_nDCG,
                shap_accuracy,
                shap_explanation_time,
                shap_explanation,
            ) = explanation(
                model,
                "SHAP",
                data,
                samples,
                expected_explanation,
                saving_path,
                experiment_name,
            )
            logger.info(f"Shap feature importance NDCG: {shap_nDCG.mean()}")

            metric_df["shap_explanation_accuracy"] = shap_accuracy
            metric_df["shap_feature_importance_ndcg"] = shap_nDCG
            metric_df["shap_explanation_time"] = shap_explanation_time

        if "grad_explanation_accuracy" not in existing_columns or force_rerun:
            (
                grad_nDCG,
                grad_accuracy,
                grad_explanation_time,
                grad_explanation,
            ) = explanation(
                model,
                "grad",
                data,
                samples,
                expected_explanation,
                saving_path,
                experiment_name,
            )
            logger.info(f"Grad feature importance NDCG: {grad_nDCG.mean()}")
            metric_df["grad_explanation_accuracy"] = grad_accuracy
            metric_df["grad_explanation_time"] = grad_explanation_time
            metric_df["grad_ndcg"] = grad_nDCG

    elif cfg.model.model_name == "DDPM":
        if (
            "reconstruction_error_accuracy" not in existing_columns
            or force_rerun
        ):
            (
                reconstruct_error_nDCG,
                reconstruct_error_accuracy,
                reconstruct_error_time,
                reconstruct_error,
            ) = explanation(
                model,
                method="reconstruction_error",
                data=data,
                samples=samples,
                expected_explanation=expected_explanation,
                saving_path=saving_path,
                exp_name=experiment_name,
            )
            metric_df["reconstruction_error_accuracy"] = (
                reconstruct_error_accuracy
            )
            metric_df["reconstruction_error_ndcg"] = reconstruct_error_nDCG
            metric_df["reconstruction_error_time"] = reconstruct_error_time

        if "shap_explanation_accuracy" not in existing_columns or force_rerun:
            (
                shap_nDCG,
                shap_accuracy,
                shap_explanation_time,
                shap_explanation,
            ) = explanation(
                model,
                "SHAP",
                data,
                samples,
                expected_explanation,
                saving_path,
                experiment_name,
            )
            metric_df["shap_explanation_accuracy"] = shap_accuracy
            metric_df["shap_feature_importance_ndcg"] = shap_nDCG
            metric_df["shap_explanation_time"] = shap_explanation_time

    metric_df.to_csv(Path(saving_path, "model_metrics.csv"), index=False)
    # Dump used configuration
    omegaconf.OmegaConf.save(cfg, Path(saving_path, "experiment_config.yaml"))
    logger.info(f"Everything saved in {saving_path}")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config_explainability",
)
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger(__name__)
    device = check_cuda(logger, cfg.device)
    if cfg.mode == "benchmark":
        if cfg.training_method.name == "DSIL":
            for ratio in [0.5, "cosine", "exponential"]:
                cfg.training_method.ratio = ratio
                for sampling_method in ["deterministic"]:
                    cfg.training_method.sampling_method = sampling_method
                    run_config(cfg, logger, device)
        else:
            run_config(cfg, logger, device)
    elif cfg.mode == "debug":
        cfg.model.training.epochs = 3
        if cfg.training_method.name == "DSIL":
            cfg.training_method.max_iter = 2
        run_config(cfg, logger, device)


if __name__ == "__main__":
    main()
