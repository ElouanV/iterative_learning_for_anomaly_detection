import logging
import time
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import sklearn.metrics as skm
from adbench.myutils import Utils
from matplotlib import pyplot as plt

from train_model import train_model
from utils import (check_cuda, get_dataset, low_density_anomalies,
                   select_model, setup_experiment, train_model)
from viz.training_viz import (plot_anomaly_score_distribution,
                              plot_anomaly_score_distribution_split, plot_tsne)


def run_config(cfg, logger, device):
    utils = Utils()  # utils function
    utils.set_seed(cfg.random_seed)
    saving_path, experiment_name = setup_experiment(cfg)
    data = get_dataset(cfg)
    logger.info(
        f"Model name: {cfg.model.model_name}, dataset name: {cfg.dataset.dataset_name}, training method:"
        f" {cfg.training_method.name}, sampling name(if applicable):"
        f" {cfg.training_method.sampling_method}, ratio(for iterative leaning"
        f" sampling method only): "
        f"{cfg.training_method.ratio if cfg.training_method =='iterative_dataset_sampling' else None}, random seed:   \
            {cfg.random_seed}"
    )

    model = select_model(cfg.model, device=device)

    # training, for unsupervised models the y label will be discarded
    start_time = time.time()

    data["y_train"][data["y_train"] > 0] = 1
    data["binary_y_test"] = data["y_test"].copy()
    data["binary_y_test"][data["binary_y_test"] > 0] = 1
    model, train_log = train_model(
        cfg,
        model,
        data["X_train"],
        data["y_train"],
        X_eval=data["X_test"],
        y_eval=data["binary_y_test"],
        exp_name=experiment_name,
        saving_path=saving_path,
    )
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    score = model.predict_score(data["X_test"]).squeeze()
    end_time = time.time()
    inference_time = end_time - start_time

    y_test_anomaly = data["y_test"].copy()
    y_test_anomaly[y_test_anomaly > 0] = 1
    indices = np.arange(len(data["y_test"]))
    # Suppose we know how much anomaly are in the dataset
    y_pred = low_density_anomalies(-score, len(indices[y_test_anomaly == 1]))
    f1_score = skm.f1_score(y_test_anomaly, y_pred)
    plot_tsne(
        data,
        y_test_anomaly,
        y_pred,
        exp_name=experiment_name,
        saving_path=saving_path,
    )
    threshold = np.percentile(score, y_test_anomaly.mean() * 100)
    plot_anomaly_score_distribution(
        score, saving_path, exp_name=experiment_name, threshold=threshold
    )
    plot_anomaly_score_distribution_split(
        score,
        y_test_anomaly,
        p=y_pred,
        exp_name=experiment_name,
        saving_path=saving_path,
    )

    logger.info(f"F1 score: {f1_score}")

    inds = np.where(np.isnan(score))
    score[inds] = 0

    result = utils.metric(y_true=y_test_anomaly, y_score=score)
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

    train_log = pd.DataFrame(train_log)
    if cfg.mode != "debug":
        metric_df.to_csv(
            Path(
                saving_path,
                "model_metrics.csv",
            )
        )
        train_log.to_csv(
            Path(
                saving_path,
                "train_log.csv",
            )
        )
        model.save_model(Path(saving_path, "model.pth"))
    # Dump used configuration
    omegaconf.OmegaConf.save(cfg, Path(saving_path, "experiment_config.yaml"))
    logger.info(f"Everything saved in {saving_path}")


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
            for ratio in [0.5]:
                cfg.training_method.ratio = ratio
                for sampling_method in ["deterministic"]:
                    cfg.training_method.sampling_method = sampling_method
                    if cfg.model.model_name == "DTEC":
                        for T in [100, 200, 400]:
                            cfg.model.model_parameters.T = T
                            for nb_bin in [7, 16]:
                                cfg.model.model_parameters.num_bins = nb_bin
                                run_config(cfg, logger, device)
        elif cfg.training_method.name == "unsupervised":
            for T in [100, 200, 400]:
                cfg.model.model_parameters.T = T
                for nb_bin in [7, 16, 32]:
                    cfg.model.model_parameters.num_bins = nb_bin
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
