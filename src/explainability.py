import logging

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import StandardScaler


def selective_noise(model, X):
    """
    Takes :
    |   model : trained model to evaluate
    |   X : the datas
    Return :
    |   Scores per feature using selective noise
    Description :
    |   Apply a selective noise to every feature and couple of feature
    |   then use the model to predict diffusion time t_pred and comparing it with the
    |   diffusion time predicted without noising the data.
    |   Datas are noised through the diffusion process and the time is predicted for each 10 step,
    |   then the metric use to get a score for each feature is the mean.
    """
    logger = logging.getLogger("selective_noise()")
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    indices = np.arange(len(X))
    X_scale = X.iloc[indices, :]
    scaler = StandardScaler().fit(X_scale)
    X_scaled = scaler.transform(X_scale)

    logger.info("selective noise par feature")
    err = []
    t0 = model.predict_score(X_scaled)
    # variables individuelles
    for i in range(X.shape[-1]):
        err_i = []
        for t in range(0, model.T, 10):
            X_noisy = np.copy(X_scaled)
            X_noisy[:, i] = (
                model.forward_noise(
                    torch.tensor(X_scaled[:, i]),
                    torch.tensor([t], dtype=torch.long),
                )
                .detach()
                .cpu()
                .numpy()
            )
            t_pred = model.predict_score(X_noisy)
            err_i.append(t_pred - t0)
            print(
                f"Progression: i:{i+1}/{X.shape[-1]} , t:{t+1}/{model.T}",
                end="\r",
            )
        err_i = np.array(err_i)
        err.append(err_i)
    metric = np.mean
    score_per_feature = [metric(err_i.flatten()) for err_i in err]

    logger.info("selective noise par couple")
    couple_err = np.zeros((X.shape[-1], X.shape[-1]))
    for i in range(X.shape[-1]):
        for j in range(X.shape[-1]):
            err_t = []
            for t in range(0, model.T, 10):
                X_noisy = np.copy(X_scaled)
                cols = X_scaled[:, [i, j]].T
                cols = (
                    model.forward_noise(
                        torch.tensor(cols),
                        torch.tensor([t, t], dtype=torch.long),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                if j != i:
                    X_noisy[:, i], X_noisy[:, j] = cols[0], cols[1]
                else:
                    X_noisy[:, i] = cols[0]
                t_pred = model.predict_score(X_noisy)
                err_t.append(np.mean(t_pred - t0))
                print(
                    f"Progression: i:{i+1}/{X.shape[-1]}, j:{j+1}/{X.shape[-1]} , t:{t+1}/{model.T}",
                    end="\r",
                )
            couple_err[i, j] = np.mean(err_t)

    return score_per_feature, couple_err


def shap_importances_per_feature(model, X):
    """
    takes :
    |   model : trained model
    |   X : datas
    returns :
    |   importances : list of score explaining the role played by every feature on the prediction with model
    """
    X100 = shap.utils.sample(X, 100)
    explainer = shap.Explainer(model.predict_score, X100)
    shap_values = explainer(X)

    return shap_values
