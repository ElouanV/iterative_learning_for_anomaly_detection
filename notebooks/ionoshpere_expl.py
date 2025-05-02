import os
import sys

sys.path.append("../src")
from utils import get_dataset, low_density_anomalies, select_model

import pandas as pd
from sklearn.metrics import mean_squared_error

import pandas as pd
import warnings

from pathlib import Path
# Take on random sample from the dataset
import random
from adbench.myutils import Utils
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
import shap
from shap import KernelExplainer
from tqdm.notebook import tqdm
import torch
import yaml
from hydra import compose, initialize
from scipy.stats import pearsonr

experiment_path = "../results/epoch_budget/DTEC_DSIL_deterministic_0.5_s0_T400_bins7/18_Ionosphere"
experiment_config = os.path.join(experiment_path, "experiment_config.yaml")
dataset_name = experiment_path.split("/")[-1]


def load_model_and_dataset_from_path(experiment_path, X):
    cfg_experiment_path = Path(f"{experiment_path}/experiment_config.yaml")
    model_path = f"{experiment_path}/model.pth"

    with initialize(config_path=str(experiment_path), version_base=None):
        cfg = compose(config_name=cfg_experiment_path.name)

    model = select_model(
        cfg.model, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.load_model(model_path, X)
    return model, cfg


def load_dataset(experiment_path):
    cfg_experiment_path = Path(f"{experiment_path}/experiment_config.yaml")
    # Load as yaml
    with open(cfg_experiment_path, "r") as stream:
        config = yaml.safe_load(stream)
        if "dataset_path" in config["dataset"].keys():
            config["dataset"]["dataset_path"] = (
                "../" + config["dataset"]["dataset_path"]
            )
        else:
            config["dataset"]["dataset_path"] = (
                "../" + config["dataset"]["data_path"]
            )

    with initialize(config_path=str(experiment_path), version_base=None):
        cfg = compose(config_name=cfg_experiment_path.name)
    dataset = get_dataset(cfg)
    return dataset


dataset = load_dataset(experiment_path)
model, cfg = load_model_and_dataset_from_path(
    experiment_path, dataset["X_train"]
)


# Get anomaly score of every samples
def get_anomaly_score(model, X):
    model.model.eval()
    with torch.no_grad():
        X = torch.from_numpy(X).float()
        X = X.to(model.device)
        anomaly_score = model.predict_score(X)
    return anomaly_score


anomaly_scores = get_anomaly_score(model, dataset["X_test"])
y_pred = low_density_anomalies(
    -anomaly_scores, num_anomalies=dataset["y_test"].sum()
)
# Get intersection between prediction and ground truth, both are 1 array of 0 and 1
true_positive_indices = (y_pred == 1) & (dataset["y_test"] == 1)
data_to_explain = dataset["X_test"][true_positive_indices]


utils = Utils()
f1_score = f1_score(dataset["y_test"], y_pred)
result = utils.metric(y_true=dataset["y_test"], y_score=anomaly_scores)
print(f'AUCROC: {result["aucroc"]}')


explainer = KernelExplainer(
    model.predict_score,
    data=np.zeros((1, dataset["X_test"].shape[1])),
    silent=True,
)
shap_expl = explainer.shap_values(
    data_to_explain, nsamples=50, show_progress=False, silent=True
)

step = 20
w_explanations = np.array(
    model.instance_explanation(data_to_explain, agg="weighted_mean", step=step)
)
mean_explanations = np.array(
    model.instance_explanation(data_to_explain, agg="mean", step=step)
)
max_explanations = np.array(
    model.instance_explanation(data_to_explain, agg="max", step=step)
)

shap_expl = np.array(shap_expl)


def infidelity(x, model, attribution, num_samples=50, delta_std=0.01):
    x = x.reshape(1, -1)
    attribution = attribution.reshape(1, -1)
    # Scale so that the sum is 1
    attribution = attribution / np.sum(attribution)
    n_features = x.shape[1]
    f_x = model.predict_score(x)
    infidelities = []
    for _ in range(num_samples):
        delta = np.random.normal(0, delta_std, size=(1, n_features))
        f_x_delta = model.predict_score(x - delta)
        dot = np.dot(delta.T, attribution)
        error = (dot - f_x + f_x_delta) ** 2
        infidelities.append(error)
    return np.mean(infidelities)


def faithfulness(f, g, x, x_baseline, subset_size=5, num_samples=100):
    d = x.shape[0]
    attributions = g(x)  # Vector of shape (d,)

    attr_sums = []
    output_diffs = []

    for _ in range(num_samples):
        # Randomly select a subset of feature indices
        S = np.random.choice(d, size=subset_size, replace=False)

        # Attribution sum over S
        sum_attr = np.sum(attributions[S])

        # Create x with features in S replaced by baseline values
        x_masked = x.copy()
        x_masked[S] = x_baseline[S]

        # Difference in model output
        delta_f = f(x.reshape(1, -1)) - f(x_masked.reshape(1, -1))

        attr_sums.append(sum_attr)
        output_diffs.append(delta_f)
    attr_sums = np.array(attr_sums)
    output_diffs = np.array(output_diffs)
    # Compute Pearson correlation between attribution sums and output differences
    corr, _ = pearsonr(attr_sums, output_diffs.squeeze())
    # Replace NaN with 0
    if np.isnan(corr):
        corr = 0
    return corr


x_baseline = np.zeros((dataset["X_test"].shape[1]))

faithfulness_df = pd.DataFrame()

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
for k in range(3, dataset["X_train"].shape[1], 2):
    result = {}
    for explanation_method in ["shap", "mean", "max", "our"]:
        if explanation_method == "shap":
            g = lambda x: explainer.shap_values(
                x, nsamples=50, show_progress=False
            )
        elif explanation_method == "mean":
            g = lambda x: model.instance_explanation(x, agg="mean", step=10)
        elif explanation_method == "max":
            g = lambda x: model.instance_explanation(x, agg="max", step=10)
        elif explanation_method == "our":
            g = lambda x: model.instance_explanation(
                x, agg="weighted_mean", step=10
            )
        faithfulness_list = []
        for idx in tqdm(range(len(data_to_explain))):
            x_i = data_to_explain[idx, :]
            faithfulness_list.append(
                faithfulness(
                    model.predict_score,
                    g,
                    x_i,
                    x_baseline,
                    subset_size=k,
                    num_samples=100,
                )
            )
        faithfulness_list = np.array(faithfulness_list)
        result[explanation_method] = faithfulness_list.mean()
    faithfulness_df = pd.concat(
        [faithfulness_df, pd.DataFrame(result, index=[k])]
    )
    faithfulness_df.to_csv(os.path.join(f"faithfulness_{dataset_name}.csv"), index=True)
    print(f"k: {k}, {result}")
faithfulness_df = faithfulness_df.reset_index()

faithfulness_df.to_csv(
    os.path.join(f"faithfulness_{dataset_name}.csv"), index=False
)



def compute_fidelity_regression(
    x, f, g, train_set, top_k=5, n_samples=100, random_state=42
):
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(train_set), size=n_samples, replace=False)
    X_subset = train_set[idx]
    x = x.reshape(1, -1)

    importances = g(x).reshape(1, -1)  # shape: (n_samples, n_features)
    important_indices = np.argsort(-np.abs(importances), axis=1)[
        :, :top_k
    ]  # shape: (n_samples, top_k)
    score_orig = f(x)

    # Create the perturbed dataset
    X_subset[:, important_indices] = x[:, important_indices]
    score_perturbed = f(X_subset)
    score_orig = score_orig.repeat(n_samples, axis=0)

    fidelity_score = mean_squared_error(score_orig, score_perturbed)
    return fidelity_score


fidelity_result_pd = pd.DataFrame()
for k in range(3, dataset["X_train"].shape[1], 2):
    fidelity_result = {"k": k}
    for explanation_method in ["shap", "mean", "max", "our"]:
        if explanation_method == "shap":
            g = lambda x: explainer.shap_values(
                x, nsamples=50, show_progress=False, silent=True
            )
        elif explanation_method == "mean":
            g = lambda x: model.instance_explanation(x, agg="mean", step=10)
        elif explanation_method == "max":
            g = lambda x: model.instance_explanation(x, agg="max", step=10)
        elif explanation_method == "our":
            g = lambda x: model.instance_explanation(
                x, agg="weighted_mean", step=10
            )
        fidelity_list = []
        for idx in tqdm(range(len(data_to_explain))):
            x_i = data_to_explain[idx, :]
            fidelity_list.append(
                compute_fidelity_regression(
                    x_i,
                    model.predict_score,
                    g,
                    train_set=dataset["X_train"],
                    top_k=k,
                    n_samples=100,
                )
            )
        fidelity_list = np.array(fidelity_list)
        fidelity_result[explanation_method] = fidelity_list.mean()
    print(f"{fidelity_result}")
    fidelity_result_pd = pd.concat(
        [fidelity_result_pd, pd.DataFrame(fidelity_result, index=[0])], axis=0
    )
fidelity_result_pd = fidelity_result_pd.reset_index(drop=True)

# Save everything, infidelity, faithfulness and fidelity

fidelity_result_pd.to_csv(
    os.path.join(f"fidelity_{dataset_name}.csv"), index=False
)

print(f"Saved infidelity, faithfulness and fidelity to {dataset_name}.csv")
