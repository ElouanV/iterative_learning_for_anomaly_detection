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

from training_method.iterative_learning import SamplingIterativeLearning
from training_method.weighted_loss_iterative_learning import \
    WeightedLossIterativeLearning
from utils import (check_cuda, get_dataset, low_density_anomalies,
                   select_model, setup_experiment)
from viz.training_viz import (plot_anomaly_score_distribution,
                              plot_anomaly_score_distribution_split, plot_tsne)


def train_model(
    cfg,
    model,
    X_train,
    y_train,
    X_eval=None,
    y_eval=None,
    saving_path: Path = None,
    exp_name: str = None,
):
    """
    Train the model according to the configuration, selecting different training methods
    """
    train_log = {}
    if cfg.training_method.name == "unsupervised":
        model, train_losses = model.fit(X_train, cfg.model)
    elif cfg.training_method.name == "semi-supervised":
        model, train_losses = model.fit(
            X_train, y_train, model_config=cfg.model
        )
    elif cfg.training_method.name == "DSIL":
        iterative_learning = SamplingIterativeLearning(
            cfg,
            saving_path=saving_path,
            exp_name=exp_name,
        )
        model, train_log = iterative_learning.train(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            model=model,
            max_iter=cfg.training_method.max_iter,
        )
    elif cfg.training_method.name == "weighted_loss":
        iterative_learning = WeightedLossIterativeLearning(cfg)
        model, train_log = iterative_learning.train(
            X_train,
            y_train,
            X_eval,
            y_eval,
            model,
            cfg.iterative_learning.max_iter,
        )
    else:
        raise ValueError(f"Unknown training method: {cfg.training_method.name}")
    return model, train_log


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

    if (
        Path(saving_path, "model_metrics.csv").exists()
        and Path(saving_path, "model.pth").exists()
    ):
        logger.info("Experiment already done, skipping")
        return
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
    if cfg.mode == "benchmark":
        if cfg.training_method.name == "DSIL":
<<<<<<< HEAD
            for ratio in [0.5, "cosine", "exponential"]:
=======
            for ratio in [0.5]:
>>>>>>> 924f15f44a30f23e8f7bcbaeb04f9a2fe64116ed
                cfg.training_method.ratio = ratio
                for sampling_method in ["deterministic"]:
                    cfg.training_method.sampling_method = sampling_method
                    run_config(cfg, logger, device)
        elif cfg.training_method.name == "unsupervised":
            run_config(cfg, logger, device)
        else:
            run_config(cfg, logger, device)
    elif cfg.mode == "debug":
        cfg.model.training.epochs = 3
        if cfg.training_method.name == "DSIL":
            cfg.training_method.max_iter = 2
        run_config(cfg, logger, device)
    else:
        print(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
