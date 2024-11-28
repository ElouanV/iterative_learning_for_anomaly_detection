import logging
import time
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import sklearn.metrics as skm
from adbench.myutils import Utils

from training_method.utils import train_model
from utils import (check_cuda, get_dataset, low_density_anomalies,
                   select_model, setup_experiment)
from viz.training_viz import plot_score_density


def run_config(cfg, logger, device):
    utils = Utils()  # utils function
    utils.set_seed(cfg.random_seed)
    saving_path, experiment_name = setup_experiment(cfg)
    if Path(saving_path, "experiment_config.yaml").exists():
        logger.info(
            f"Experiment with random seed {cfg.random_seed} already exists, skipping"
        )
        return
    data = get_dataset(cfg)
    logger.info(
        f"Model name: {cfg.model.model_name}, dataset name: {cfg.dataset.dataset_name}, training method:"
        f" {cfg.training_method.name}, sampling name(if applicable):"
        f" {cfg.training_method.sampling_method}, ratio(for iterative leaning"
        f" sampling method only): "
        f"{cfg.training_method.ratio if cfg.training_method =='iterative_dataset_sampling' else None}, random seed:   \
            {cfg.random_seed}"
    )
    if cfg.model.model_name == "DeepSVDD":
        cfg.model.model_parameters = cfg.model.model_parameters[
            cfg.dataset.dataset_name
        ]
        cfg.model.model_parameters.input_dim = data["X_train"].shape[1]
    model = select_model(cfg.model, device=device)

    # training, for unsupervised models the y label will be discarded
    start_time = time.time()
    model, train_log = train_model(
        cfg,
        model,
        data["X_train"],
        data["y_train"],
        X_eval=data["X_test"],
        y_eval=data["y_test"],
        saving_path=saving_path,
        device=device,
    )
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    score = model.predict_score(data["X_test"], device=device).squeeze()
    end_time = time.time()
    inference_time = end_time - start_time

    plot_score_density(score, exp_name=experiment_name, saving_path=saving_path)
    data["y_test"][data["y_test"] > 0] = 1

    indices = np.arange(len(data["y_test"]))
    # Suppose we know how much anomaly are in the dataset
    p = low_density_anomalies(-score, len(indices[data["y_test"] == 1]))

    f1_score = skm.f1_score(data["y_test"], p)
    logger.info(f"F1 score: {f1_score}")

    inds = np.where(np.isnan(score))
    score[inds] = 0

    result = utils.metric(y_true=data["y_test"], y_score=score)
    logger.info(f'AUCROC: {result["aucroc"]}')

    metric_df = {}
    metric_df["training_time"] = training_time
    metric_df["inference_time"] = inference_time
    metric_df["f1_score"] = f1_score
    metric_df["model_name"] = cfg.model.model_name
    metric_df["dataset_name"] = cfg.dataset.dataset_name
    metric_df["training_method"] = cfg.training_method.name
    metric_df["sampling_method"] = cfg.training_method.sampling_method
    metric_df["random_seed"] = cfg.random_seed
    metric_df["aucroc"] = result["aucroc"]
    metric_df = pd.DataFrame([metric_df])
    if cfg.mode != "debug":
        metric_df.to_csv(
            Path(
                saving_path,
                "model_metrics.csv",
            )
        )
    train_log = pd.DataFrame(train_log)
    if cfg.mode != "debug":
        train_log.to_csv(
            Path(
                saving_path,
                "train_log.csv",
            )
        )
    # Dump used configuration
    omegaconf.OmegaConf.save(cfg, Path(saving_path, "experiment_config.yaml"))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger(__name__)
    device = check_cuda(logger, cfg.device)
    if cfg.dataset.data_type != "tabular":
        raise NotImplementedError(
            f"Data type {cfg.dataset.data_type} not implemented yet"
        )
    if cfg.mode == "benchmark":
        if cfg.training_method.name == "DSIL":
            for ratio in [0.5, "cosine", "exponential"]:
                cfg.training_method.ratio = ratio
                for sampling_method in ["deterministic", "probabilistic"]:
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
