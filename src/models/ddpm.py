"""
Implementation of DDPM comes from https://github.com/vicliv/DTE

Part of the code is adapted from the ResNet model (https://arxiv.org/abs/2106.11959)
provided at https://github.com/Yura52/rtdl and also adapted from https://github.com/rotot0/tab-ddpm
The model was modified to integrate Time Embedding.
On Diffusion Modeling for Anomaly Detection - Diffusion Time Estimation
@Author: Victor Livernoche <vlivernoche@gmail.com>
"""

import math
from pathlib import Path
from typing import Callable, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.losses import WMSELoss
from models.base_model import BaseModel
from viz.training_viz import plot_nmf, reconstruction_error_boxplot

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == "ReGLU"
            else (
                GEGLU()
                if module_type == "GEGLU"
                else getattr(nn, module_type)(*args)
            )
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)
                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)
          Head: (in) -> Norm -> Activation -> Linear -> (out)
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
        Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        d_emb: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_emb, d_main))

    @classmethod
    def make_baseline(
        cls: Type["ResNet"],
        *,
        d_in: int,
        d_emb: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> "ResNet":
        """Create a "baseline" `ResNet`.
        This variation of ResNet was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`
        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
            Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            d_in=d_in,
            d_emb=d_emb,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization="BatchNorm1d",
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x, time_emb) -> Tensor:
        x = x.float()
        x = self.first_layer(x)
        h = self.mlp(time_emb)
        x = x + h
        x = self.blocks(x)
        x = self.head(x)
        return x


class ResNetDiffusion(nn.Module):
    def __init__(self, d_in, num_classes, rtdl_params, dim_t=256):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes

        rtdl_params["d_in"] = d_in
        rtdl_params["d_out"] = d_in
        rtdl_params["d_emb"] = dim_t
        self.resnet = ResNet.make_baseline(**rtdl_params)

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if y is not None and self.num_classes > 0:
            emb += self.label_emb(y.squeeze())
        return self.resnet(x, emb)


class DDPM(BaseModel):
    def __init__(
        self,
        seed=0,
        model_name="DDPM",
        epochs=400,
        batch_size=32,
        lr=1e-4,
        weight_decay=5e-4,
        T=1000,
        reconstruction_t=250,
        full_path=False,
        device=None,
        resnet_parameters={},
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.full_path = full_path

        self.T = T
        self.rec_t = reconstruction_t if reconstruction_t < T else int(T * 0.25)

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.seed = seed
        self.resnet_parameters = resnet_parameters
        betas = torch.linspace(0.0001, 0.01, T + 1)  # linear beta scheduling

        # Pre-calculate different terms for closed form of diffusion process
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        def forward_noise(x_0, t, drift=True):
            """
            Takes data point and a timestep as input and
            returns the noisy version of it
            """
            noise = torch.randn_like(x_0)  # epsilon

            noise.requires_grad_()  # for the backward propagation of the NN
            sqrt_alphas_cumprod_t = (
                torch.take(sqrt_alphas_cumprod, t.cpu()).to(device).unsqueeze(1)
            )
            sqrt_one_minus_alphas_cumprod_t = (
                torch.take(sqrt_one_minus_alphas_cumprod, t.cpu())
                .to(device)
                .unsqueeze(1)
            )

            # mean + variance
            if drift:
                return (
                    sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device)
                    + sqrt_one_minus_alphas_cumprod_t.to(self.device)
                    * noise.to(self.device)
                ).to(torch.float32), noise.to(self.device)
            else:  # variance only
                return (
                    x_0.to(self.device)
                    + sqrt_one_minus_alphas_cumprod_t.to(self.device)
                    * noise.to(self.device)
                ).to(torch.float32), noise.to(self.device)

        def get_loss(model, x_0, t, weights=None):
            # get the loss based on the input and timestep

            # get noisy1 - sqrt_alphas_cumprod_t sample
            x_noisy, noise = forward_noise(x_0, t)

            # predict the timestep
            noise_pred = model(x_noisy, t)

            # For the regression model, the target is t with mean squared error loss
            loss_fn = WMSELoss()

            loss = loss_fn(noise_pred, noise, weights)

            return loss

        def p_sample(model, x, t):
            t_index = t[0]
            with torch.no_grad():
                betas_t = (
                    torch.take(betas, t.cpu()).unsqueeze(1).to(self.device)
                )
                sqrt_one_minus_alphas_cumprod_t = (
                    torch.take(sqrt_one_minus_alphas_cumprod, t.cpu())
                    .unsqueeze(1)
                    .to(self.device)
                )
                sqrt_recip_alphas_t = (
                    torch.take(sqrt_recip_alphas, t.cpu())
                    .unsqueeze(1)
                    .to(self.device)
                )

                p_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
                )

                if t_index == 0:
                    sample = p_mean
                else:
                    posterior_variance_t = (
                        torch.take(posterior_variance, t.cpu())
                        .unsqueeze(1)
                        .to(self.device)
                    )

                    z = torch.randn_like(x).to(self.device)
                    sample = p_mean + torch.sqrt(posterior_variance_t) * z

            return sample

        self.forward_noise = forward_noise
        self.sample = p_sample
        self.loss_fn = get_loss
        self.model = None

    def reconstruct(self, x, t):
        with torch.no_grad():
            b = x.shape[0]
            xs = []
            x_noisy, _ = self.forward_noise(
                x, torch.full((b,), t, device=self.device).long()
            )
            x_noisy = x
            for i in reversed(range(0, t)):
                x_noisy = self.sample(
                    self.model,
                    x_noisy,
                    torch.full((b,), i, device=self.device).long(),
                )
                xs.append(x_noisy)
        return xs

    def fit(
        self,
        X_train,
        y_train=None,
        weights=None,
        model_config=None,
        device=None,
    ):
        if self.model is None:  # allows retraining

            self.model = ResNetDiffusion(
                X_train.shape[-1], 0, self.resnet_parameters
            ).to(self.device)

        optimizer = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        data = (
            X_train
            if weights is None
            else list(
                zip(
                    torch.from_numpy(X_train).float(),
                    torch.from_numpy(weights).float(),
                )
            )
        )
        train_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        train_losses = []
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            loss_ = []

            for x in train_loader:
                w = None
                if weights is not None:
                    w = x[1]
                    x = x[0]
                    w = w.to(self.device)
                x = x.to(self.device)
                optimizer.zero_grad()

                # sample t uniformly
                t = torch.randint(
                    0, self.T, (x.shape[0],), device=self.device
                ).long()

                # compute the loss
                loss = self.loss_fn(self.model, x, t, w)

                loss.backward()
                optimizer.step()
                loss_.append(loss.item())

            train_losses.append(np.mean(np.array(loss_)))
            if epoch > 50:
                if (
                    train_losses[len(train_losses) - 1]
                    > train_losses[len(train_losses) - 40]
                ):
                    break
        return self, train_losses

    @torch.no_grad()
    def predict_score(self, X, reconstruction_t=None, device=None):
        test_loader = DataLoader(
            torch.from_numpy(X).float(),
            batch_size=1000,
            shuffle=False,
            drop_last=False,
        )
        preds = []
        self.model.eval()
        if reconstruction_t is not None:
            self.rec_t = reconstruction_t
        for x in tqdm(test_loader, desc="Predicting"):
            x = x.to(self.device)
            # predict the timestep based on x, or the probability of each class for the classification
            x_rec = self.reconstruct(x, self.T)

            if not self.full_path:
                pred = (
                    ((x - x_rec[-1]) ** 2)
                    .mean(1)
                    .squeeze()
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                prev = x
                total = np.zeros((x.shape[0],))
                for rec in x_rec:
                    pred = ((prev - rec) ** 2).mean(1)
                    total += pred.squeeze().cpu().detach().numpy()
                    prev = rec
                pred = total

            preds.append(pred)
        # If preds contains multiple batches, concatenate them
        if len(preds) > 1:
            preds = np.concatenate(preds, axis=0)
        else:
            preds = np.array(preds)
        preds = preds.reshape(-1, 1)
        return preds

    @torch.no_grad()
    def raw_prediction(self, x, reconstruction_t=None):
        self.model.eval()
        if reconstruction_t is not None:
            self.rec_t = reconstruction_t
            # predict the timestep based on x, or the probability of each class for the classification
        return self.reconstruct(x, self.rec_t)

    def feature_wise_explanation(
        self, X, y, saving_path, bath_size=32, n_components=2
    ):

        # For each data of the eval set, compute reconstruction error
        X_anomaly = X[y == 1]
        anomaly_eval_data_loader = DataLoader(
            X_anomaly,
            batch_size=bath_size,
            shuffle=False,
        )
        X_normal = X[y == 0]
        normal_eval_data_loader = DataLoader(
            X_normal,
            batch_size=bath_size,
            shuffle=False,
        )
        anomaly_reconstruction_errors = []
        for _, (X) in tqdm(enumerate(anomaly_eval_data_loader), desc="Anomaly"):
            X = X.to(self.device)
            reconstructed_x = self.raw_prediction(X)[-1]
            reconstructed_x = reconstructed_x.to("cpu")
            X = X.to("cpu")
            # Compute reconstruction error
            error = torch.abs(X - reconstructed_x)
            anomaly_reconstruction_errors.append(error)
        normal_reconstruction_errors = []
        for _, (X) in tqdm(enumerate(normal_eval_data_loader), desc="Normal"):
            X = X.to(self.device)
            reconstructed_x = self.raw_prediction(X)[-1]
            reconstructed_x = reconstructed_x.to("cpu")
            X = X.to("cpu")
            # Compute reconstruction error
            error = torch.abs(X - reconstructed_x)
            normal_reconstruction_errors.append(error)

        anomaly_reconstruction_errors = torch.concat(
            anomaly_reconstruction_errors, dim=0
        )

        normal_reconstruction_errors = torch.concat(
            normal_reconstruction_errors, dim=0
        )
        w, h = self.non_negative_matrix_factorization(
            anomaly_reconstruction_errors, n_components=n_components
        )

        return anomaly_reconstruction_errors, normal_reconstruction_errors, w, h

    @staticmethod
    def non_negative_matrix_factorization(v, n_components=2):
        nmf = NMF(
            n_components=n_components,
            init="random",
            random_state=0,
            max_iter=10000,
        )
        w = nmf.fit_transform(v)
        h = nmf.components_
        return w, h

    def instance_explanation(
        self,
        x,
        expected_explanation,
        saving_path,
        exp_name,
        plot=True,
        **kwargs,
    ):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = torch.from_numpy(x).float().to(self.device)
        reconstructed_x = self.raw_prediction(x)[-1]
        reconstructed_x = reconstructed_x.to("cpu")
        x = x.to("cpu")
        error = torch.abs(x - reconstructed_x)
        if plot:
            for i in range(x.shape[0]):
                plt.figure()

                scores = np.abs(error[i].numpy())
                scores = scores / np.sum(scores)
                plt.bar(
                    np.arange(x.shape[1]),
                    scores,
                    label="Reconstruction error",
                )
                plt.bar(
                    np.arange(x.shape[1]),
                    expected_explanation[i],
                    alpha=0.4,
                    label="Expected explanation",
                )
                plt.xlabel("Feature")
                plt.ylabel("Reconstruction error")
                plt.legend()
                plt.title(f"Reconstruction error for instance {i}, {exp_name}")
                plt.savefig(
                    Path(
                        saving_path,
                        f"{exp_name}_instance_{i}_reconstruction_error.png",
                    )
                )
                plt.close()
        return error

    def global_explanation(
        self, X, y_pred, experiment_name, saving_path, plot=True, **kwargs
    ):
        ncomponents = kwargs.get("n_components", 2)
        anomaly_reconstruction_error, normal_reconstruction_error, w, h = (
            self.feature_wise_explanation(
                X=X,
                y=y_pred,
                saving_path=saving_path,
                n_components=ncomponents,
            )
        )
        if plot:
            reconstruction_error_boxplot(
                anomaly_reconstruction_error,
                exp_name=experiment_name,
                saving_path=saving_path,
            )
            plot_nmf(w, h, exp_name=experiment_name, saving_path=saving_path)
        np.save(
            Path(saving_path, "anomaly_reconstruction_error.npy"),
            anomaly_reconstruction_error,
        )
        np.save(
            Path(saving_path, "normal_reconstruction_error.npy"),
            normal_reconstruction_error,
        )
        np.save(Path(saving_path, "NMF_w.npy"), w)
        np.save(Path(saving_path, "NMF_h.npy"), h)

        return anomaly_reconstruction_error, normal_reconstruction_error, w, h

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        return self
