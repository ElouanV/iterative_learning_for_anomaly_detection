import logging
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from training_method.iterative_learning import SamplingIterativeLearning
from training_method.weighted_loss_iterative_learning import \
    WeightedLossIterativeLearning
from utils import configure_logger, get_dataset, pred_from_scores, select_model


def train_model(
    cfg,
    model,
    X_train,
    y_train,
    X_eval=None,
    y_eval=None,
    saving_path: Path = None,
):
    train_log = {}
    if cfg.training_method.name == "unsupervised":
        model.fit(X_train, cfg.model)
    elif cfg.training_method.name == "semi-supervised":
        model.fit(X_train, y_train, cfg.model)
    elif cfg.training_method.name == "dataset_sampling":
        iterative_learning = SamplingIterativeLearning(
            cfg, saving_path=saving_path
        )
        model, train_log = iterative_learning.train(
            X_train,
            y_train,
            X_eval,
            y_eval,
            model,
            cfg.training_method.max_iter,
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: omegaconf.DictConfig):
    experiment_name = f"{cfg.model.model_name}_{cfg.training_method.name}_{cfg.training_method.sampling_method}"
    saving_path = Path(
        cfg.output_path, experiment_name, cfg.dataset.dataset_name
    )
    saving_path.mkdir(parents=True, exist_ok=True)

    configure_logger(saving_path)
    logger = logging.getLogger(__name__)

    os.makedirs(cfg.output_path, exist_ok=True)
    model = select_model(cfg.model)

    # FIXME: Random seed for reproducibility
    seed = cfg.random_seed

    X, y = get_dataset(Path(cfg.dataset.dataset_path))
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.dataset.test_ratio, random_state=seed, stratify=y
    )
    model_log = {"f1_score": [], "roc_auc_score": []}

    logger.info(
        f"Training model {cfg.model.model_name} on dataset {cfg.dataset.dataset_name}"
    )

    model, train_log = train_model(
        cfg, model, X_train, y_train, X_test, y_test, saving_path
    )

    scores = model.predict_score(X_test)
    nb_anomalies = np.sum(y_test)
    y_pred = pred_from_scores(scores, nb_anomalies)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"F1 score after iterative learning: {f1}")
    roc = roc_auc_score(y_test, y_pred)
    logger.info(f"ROC AUC score after iterative learning: {roc}")
    model_log["f1_score"] = f1
    model_log["roc_auc_score"] = roc
    metric_df = pd.DataFrame(model_log, index=[0])

    metric_df.to_csv(
        Path(
            cfg.output_path,
            experiment_name,
            cfg.dataset.dataset_name,
            "model_metrics.csv",
        )
    )
    # Dump used configuration
    omegaconf.OmegaConf.save(cfg, Path(saving_path, "experiment_config.yaml"))


if __name__ == "__main__":
    import torch

    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"Cuda version: {torch.version.cuda}")
    else:
        warnings.warn("Cuda not available")

    # Disable tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    main()
