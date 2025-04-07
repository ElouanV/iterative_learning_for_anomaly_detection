# This code is from https://github.com/vicliv/DTE by (V. Livernoche, V. Jain, Y. Hezaveh, S. Ravanbakhsh)


import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.losses import WMSELoss, WCrossEntropyLoss
from viz.training_viz import plot_feature_importance


class MLP(nn.Module):
    def __init__(self, hidden_sizes, num_bins=7):
        super().__init__()
        self.hidden_sizes = hidden_sizes  # hidden layers sizes
        self.activation = nn.ReLU()  # activation to use in the network

        layers = []
        for i in range(1, len(self.hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        if num_bins > 1:
            # if we have the classification model
            layers.append(nn.Linear(hidden_sizes[-1], num_bins))
            self.softmax = nn.Softmax(dim=1)
        else:
            # if we have the regression model
            layers.append(nn.Linear(hidden_sizes[-1], 1))
            self.softmax = lambda x: x  # ignore softmaxt

        self.layers = nn.ModuleList(layers)

        self.drop = torch.nn.Dropout(p=0.5, inplace=False)  # dropout

    def forward(self, x):
        x = self.activation(self.layers[0](x))

        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
            x = self.drop(x)

        return self.softmax(self.layers[-1](x))


def binning(t, T=300, num_bins=30, device="cpu"):
    """
    Gives the bin number for a given t based on T (maximum) and the number of bins
    This is floor(t*num_bins/T) bounded by 0 and T-1
    """
    return torch.maximum(
        torch.minimum(
            torch.floor(t * num_bins / T).to(device),
            torch.tensor(num_bins - 1).to(device),
        ),
        torch.tensor(0).to(device),
    ).long()


class DTE:
    def __init__(
        self,
        seed=0,
        model_name="DTE",
        hidden_size=[256, 512, 256],
        epochs=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=5e-4,
        T=400,
        num_bins=7,
        device=None,
    ):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.T = T
        self.num_bins = num_bins

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        self.seed = seed

        betas = torch.linspace(0.0001, 0.01, T)  # linear beta scheduling

        # Pre-calculate different terms for closed form of diffusion process
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod = alphas_cumprod

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        def forward_noise(x_0, t, drift=False):
            """
            Takes data point and a timestep as input and
            returns the noisy version of it
            """
            noise = torch.randn_like(x_0)  # epsilon

            noise.requires_grad_()  # for the backward propagation of the NN
            sqrt_alphas_cumprod_t = (
                torch.take(sqrt_alphas_cumprod, t.cpu())
                .to(self.device)
                .unsqueeze(1)
            )
            sqrt_one_minus_alphas_cumprod_t = (
                torch.take(sqrt_one_minus_alphas_cumprod, t.cpu())
                .to(self.device)
                .unsqueeze(1)
            )

            # mean + variance
            if drift:
                return (
                    sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device)
                    + sqrt_one_minus_alphas_cumprod_t.to(self.device)
                    * noise.to(self.device)
                ).to(torch.float32)
            else:  # variance only
                return (
                    x_0.to(self.device)
                    + sqrt_one_minus_alphas_cumprod_t.to(self.device)
                    * noise.to(self.device)
                ).to(torch.float32)

        self.forward_noise = forward_noise
        self.model = None
        self.logger = logging.getLogger(model_name)

    def compute_loss(self, x, t):
        pass

    def create_model(self, X_train):
        self.model = MLP(
                [X_train.shape[-1]] + self.hidden_size, num_bins=self.num_bins
            ).to(self.device)
        return self.model

    def fit(
        self,
        X_train,
        y_train=None,
        X_test=None,
        y_test=None,
        verbose=True,
        weights=None,
        model_config=None,
        device="cuda",
    ):
        if self.model is None:  # allows retraining
            self.model = self.create_model(X_train)

        optimizer = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        class CustomDataset(Dataset):
            def __init__(self, data, weights=None):
                self.data = data
                self.weights = weights

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                sample = self.data[idx]
                if self.weights is not None:
                    weight = self.weights[idx]
                    return sample, weight
                return sample
        dataset = CustomDataset(X_train, weights)

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        train_losses = []
        with tqdm(total=self.epochs, desc="Epochs") as pbar:
            for _ in range(self.epochs):
                self.model.train()
                loss_ = []
                for batch in train_loader:
                    if isinstance(batch, list):
                        x, w = batch
                        x = x.to(self.device)
                        w = w.to(self.device)
                    else:
                        x = batch.to(self.device)
                        w = None
                    optimizer.zero_grad()

                    # sample t uniformly
                    t = torch.randint(
                        0, self.T, (x.shape[0],), device=self.device
                    ).long()

                    # compute the loss
                    loss = self.compute_loss(x, t, w)

                    loss.backward()
                    optimizer.step()
                    loss_.append(loss.item())
                train_losses.append(np.mean(np.array(loss_)))
                pbar.update(1)
                pbar.set_postfix({"Loss: ": "{:.4f}".format(train_losses[-1])})
        return self, train_losses

    def predict_score(self, X, give_preds_binned=False, device="cuda"):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        test_loader = DataLoader(
            X,
            batch_size=100,
            shuffle=False,
            drop_last=False,
        )
        preds = []
        self.model.eval()
        for x in test_loader:
            # predict the timestep based on x, or the probability of each class for the classification
            pred_t = self.model(x.to(self.device).to(torch.float32))
            preds.append(pred_t.cpu().detach().numpy())
        preds = np.concatenate(preds, axis=0)

        if self.num_bins > 1:
            preds = np.matmul(preds, np.arange(0, preds.shape[-1]))
        else:
            preds = preds.squeeze()
        if give_preds_binned:
            return preds, binning(preds, T=self.T, num_bins=self.num_bins)
        return preds

    def explain(self, X, y=None, device="cuda", saving_path=None, step=10):

        err = []
        t0 = self.predict_score(X)
        # variables individuelles
        for i in tqdm(range(X.shape[-1]), desc="Single features"):
            err_i = []
            for t in range(0, self.T, step):
                X_noisy = np.copy(X)
                X_noisy[:, i] = (
                    self.forward_noise(
                        torch.tensor(X[:, i]),
                        torch.tensor([t], dtype=torch.long),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                t_pred = self.predict_score(X_noisy)
                err_i.append(np.abs(t_pred - t0))
            err_i = np.array(err_i)

            err_i = np.mean(err_i)

            err.append(err_i)
        return err


class DTECategorical(DTE):
    def __init__(
        self,
        seed=0,
        model_name="DTE_categorical",
        hidden_size=[256, 512, 256],
        epochs=400,
        batch_size=64,
        lr=1e-4,
        weight_decay=5e-4,
        T=400,
        num_bins=7,
        device=None,
    ):
        if num_bins < 2:
            raise ValueError("num_bins must be greater than or equal to 2")

        super().__init__(
            seed,
            model_name,
            hidden_size,
            epochs,
            batch_size,
            lr,
            weight_decay,
            T,
            num_bins,
            device=device,
        )

    def compute_loss(self, x_0, t, weights=None):
        # get the loss based on the input and timestep

        # get noisy sample
        x_noisy = self.forward_noise(x_0, t)

        # predict the timestep
        t_pred = self.model(x_noisy)

        # For the categorical model, the target is the binned t with cross entropy loss
        target = binning(
            t, T=self.T, device=self.device, num_bins=self.num_bins
        )
        if weights is None:
            loss = nn.CrossEntropyLoss()(t_pred, target)
        else: 
            loss = WCrossEntropyLoss()(t_pred, target, weights)

        return loss

    def instance_explanation(
        self,
        x,
        step=10,
        saving_path=None,
        agg="mean",
    ):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        nb_samples, nb_features = x.shape
        err = np.zeros((nb_samples, nb_features))
        if agg == "max":
            arg_max = []
        t0 = self.predict_score(x)
        for i in range(x.shape[-1]):
            err_i = []
            for t in range(0, self.T, step):
                X_noisy = np.copy(x)
                X_noisy[:, i] = (
                    self.forward_noise(
                        torch.tensor(x[:, i]),
                        torch.tensor([t], dtype=torch.long),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                t_pred = self.predict_score(X_noisy)
                err_i.append(np.abs(t_pred - t0))

            # Compute the mean error across timesteps for this feature
            err_i = np.array(err_i)  # Shape: (num_timesteps, nb_samples)
            if agg == "mean":
                err_i = np.mean(err_i, axis=0)
            elif agg == "max":
                arg_max_i = np.argmax(err_i, axis=0)
                arg_max.append(arg_max_i)
                err_i = np.max(err_i, axis=0)
            err[:, i] = err_i
        if agg == "max" and saving_path:
            np.save(Path(saving_path, "arg_max.npy"), arg_max)
        return np.array(err).squeeze()

    def gradient_explanation(self, x):
        """
        Compute the gradient of the model with respect to the input to compute feature importance vector for each sample
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        feature_importance = []
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        data_loader = DataLoader(
            x,
            batch_size=100,
            shuffle=False,
            drop_last=False,
        )
        for i, x in enumerate(data_loader):
            x = x.to(self.device)
            x.requires_grad = True
            self.model.zero_grad()
            t_pred = self.model(x)
            t_pred.backward(t_pred)
            gradient_batch = x.grad.cpu().detach().numpy()
            feature_importance.append(gradient_batch)
        feature_importance = np.concatenate(feature_importance, axis=0)
        # Normalize the gradient
        feature_importance = torch.from_numpy(feature_importance)
        feature_importance = torch.nn.functional.softmax(
            feature_importance, dim=1
        )
        return feature_importance.squeeze()

    def global_explanation(
        self,
        X,
        y_pred,
        experiment_name,
        saving_path,
        plot=True,
        step=10,
        **kwargs,
    ):
        X = X[y_pred == 1]
        feature_score = self.explain(X, step=step)
        if plot:
            plot_feature_importance(
                feature_score, exp_name=experiment_name, saving_path=saving_path
            )
            # plot_couple_feature_importance_matrix(
            #     couple_feature_score,
            #     exp_name=experiment_name,
            #     saving_path=saving_path,
            # )

        np.save(Path(saving_path, "feature_score.npy"), feature_score)
        # np.save(
        #     Path(saving_path, "couple_feature_score.npy"), couple_feature_score
        # )
        return feature_score  # couple_feature_score

    def save_model(self, path):
        self.logger.info(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)

    def load_model(self, path, X):
        self.model = MLP(
            [X.shape[-1]] + self.hidden_size, num_bins=self.num_bins
        ).to(self.device)
        self.model.load_state_dict(torch.load(path))


class DTEInverseGamma(DTE):
    def __init__(
        self,
        seed=0,
        model_name="DTE_inverse_gamma",
        hidden_size=[256, 512, 256],
        epochs=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=5e-4,
        T=400,
    ):
        super().__init__(
            seed,
            model_name,
            hidden_size,
            epochs,
            batch_size,
            lr,
            weight_decay,
            T,
            num_bins=0,
        )

    def compute_loss(self, x_0, t, weights=None):
        # get the loss based on the input and timestep
        _, dim = x_0.shape
        eps = 1e-5
        # get noisy sample
        x_noisy = self.forward_noise(x_0, t)

        # predict the inv gamma parameter
        sqrt_beta_pred = self.model(x_noisy)
        beta_pred = torch.pow(sqrt_beta_pred, 2).squeeze()

        var_target = (1.0 - self.alphas_cumprod[t.cpu()]).to(self.device)
        log_likelihood = (0.5 * dim - 1) * torch.log(
            beta_pred + eps
        ) - beta_pred / (var_target)
        log_likelihood = log_likelihood * weights
        loss = -log_likelihood.mean()

        return loss


class DTEGaussian(DTE):
    def __init__(
        self,
        seed=0,
        model_name="DTE_gaussian",
        hidden_size=[256, 512, 256],
        epochs=400,
        batch_size=64,
        lr=1e-4,
        weight_decay=5e-4,
        T=400,
    ):
        super().__init__(
            seed,
            model_name,
            hidden_size,
            epochs,
            batch_size,
            lr,
            weight_decay,
            T,
            0,
        )

    def compute_loss(self, x_0, t, weights=None):
        # get the loss based on the input and timestep

        # get noisy sample
        x_noisy = self.forward_noise(x_0, t)

        # predict the timestep
        t_pred = self.model(x_noisy)

        t_pred = t_pred.squeeze()
        target = t.float()

        loss = WMSELoss(t_pred, target, weights=weights)

        return loss


class DTEBagging:
    def __init__(
        self,
        num_bags=5,
        hidden_size=[256, 512, 256],
        epochs=200,
        batch_size=64,
        lr=1e-4,
        weight_decay=5e-4,
        T=300,
        num_bins=7,
    ):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.T = T
        self.num_bins = num_bins

        self.num_bags = num_bags

        self.models = []

    def fit(
        self,
        X_train,
        y_train=None,
        X_test=None,
        Y_test=None,
        weights=None,
        model_config=None,
        device=None,
    ):
        for _ in range(self.num_bags):
            if self.num_bags > 1:
                indices = np.arange(len(X_train))
                # np.random.seed(self.seed)
                random_idx = np.random.choice(indices, size=len(indices))
                X_train = X_train[random_idx, :]

            model = DTECategorical(
                hidden_size=self.hidden_size,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                T=self.T,
                num_bins=self.num_bins,
            )
            self.models.append(model)

            model.fit(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=Y_test,
                weights=weights,
                model_config=model_config,
            )

        return self

    def predict_score(self, X, device="cuda"):
        total = []

        # compute prediction for all models
        for model in self.models:
            total.append(model.predict_score(X))

        # sum the predictions
        pred = np.stack(total)
        preds = np.sum(pred, axis=0)

        return preds
