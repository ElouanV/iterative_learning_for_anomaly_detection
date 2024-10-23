from pathlib import Path

from training_method.iterative_learning import SamplingIterativeLearning
from training_method.weighted_loss_iterative_learning import \
    WeightedLossIterativeLearning


def train_model(
    cfg,
    model,
    X_train,
    y_train,
    X_eval=None,
    y_eval=None,
    saving_path: Path = None,
    device: str = "cuda",
):
    train_log = {}
    if cfg.training_method.name == "unsupervised":
        model, train_losses = model.fit(
            X_train, model_config=cfg.model, device=device
        )
    elif cfg.training_method.name == "semi-supervised":
        model, train_losses = model.fit(
            X_train, y_train, model_config=cfg.model, device=device
        )
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
            device=device,
        )
    elif cfg.training_method.name == "weighted_loss":
        iterative_learning = WeightedLossIterativeLearning(
            max_round=cfg.training_method.max_round,
            epochs=cfg.model.training.epochs,
            batch_size=cfg.model.training.batch_size,
            lr=cfg.model.training.learning_rate,
            device=device,
            model=model,
            model_config=cfg.model,
        )
        model, train_log = iterative_learning.train(
            X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval
        )
    else:
        raise ValueError(f"Unknown training method: {cfg.training_method.name}")
    return model, train_log
