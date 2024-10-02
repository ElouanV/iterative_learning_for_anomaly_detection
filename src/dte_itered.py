import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn.metrics as skm
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import configure_logger, plot_f1_and_loss, select_model

# ____________________________________________________________________________
#                                                                           |
#    I) Utility functions                                                   |
# ___________________________________________________________________________|


def count_ano(indices, y):
    """
    count the number of anomalies among indices
    """
    return sum(y[indices])


def pred_from_scores(scores, num_anomalies):
    """
    retourne une prédiction y où
    y[i] = 1 si scores[i] est l'un des num_anomalies plus grand score
        et 0 sinon.
    """

    indices_sorted = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )
    result = [0] * len(scores)
    # Mettre des 1 aux p premiers indices triés
    for i in indices_sorted[:num_anomalies]:
        result[i] = 1
    return result


def get_normal_indices(
    scores,
    p,
    method="constant",
    iteration=-1,
    nu_min=50,
    nu_max=100,
    max_iter=10,
):
    """
    takes :
    |   scores : t_pred obtained by DTE model
    |   p : percentage of indices that we want to keep for training
    return :
    |   output : list of indices that are the most likely to be normal and should be use for next training phase
    """
    if method == "cosine":
        p = nu_min + 1 / 2 * (nu_max - nu_min) * (
            1 + np.cos(np.pi * iteration / max_iter)
        )
    n = scores.shape[0]
    indices_sorted = sorted(range(len(scores)), key=lambda i: scores[i])
    return indices_sorted[: int(n * p)]


def barplot(scores, xlabel="", ylabel="", title="", file_path="res/barplot"):
    """
    Takes:
    |   scores : the scores to plot (should be a score per feature)
    |   other args are explicit and used for ploting
    Description :
    |   plot the scores of the different feature on a barplot and saves it in file_path
    """

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(scores) + 1), scores, align="center")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_path)


def map_anomalies(y):
    """
    Takes :
    |   y : anomaly classes (0 if anomaly, 1 or more if there are several types of anomalies)
    Return :
    |   mappey_y : the y binary target (0 = no anomaly, 1 = anomaly)
    |   num_classes : the number of types of anomalies + 1
    """
    mapped_y = np.where(y > 0, 1, 0)
    num_classes = len(np.unique(y))
    return mapped_y, num_classes


# ____________________________________________________________________________
#                                                                           |
#    II) Semi-supervised training to see the best achievable case           |
# ___________________________________________________________________________|


def semi_supervised(X, y, dataset_name, model_name):
    logger = logging.getLogger("semi_supervised()")
    logger.info("Semi supervised evaluation")
    # Semi-supervised :

    X_normal = X[y == 0]
    y_normal = y[y == 0]
    X_anomaly = X[y != 0]
    y_anomaly = y[y != 0]

    X_train_normal, X_test_normal, y_train_normal, y_test_normal = (
        train_test_split(X_normal, y_normal, test_size=0.5, random_state=42)
    )
    X_test = np.concatenate((X_test_normal, X_anomaly), axis=0)
    y_test = np.concatenate((y_test_normal, y_anomaly), axis=0)
    X_train = X_train_normal
    _ = y_train_normal

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    semi_model = select_model(model_name)
    semi_model.fit(X_train)
    scores0 = semi_model.predict_score(X_test)
    y_pred0 = pred_from_scores(scores0, np.sum(y_test))
    F1_0 = skm.f1_score(y_test, y_pred0)

    logger.info("F1_semi_supervised :" + str(F1_0))
    return F1_0


# ____________________________________________________________________________
#                                                                           |
#   III) Iteration functions                                                |
# ___________________________________________________________________________|


def iter(
    X_train,
    X_scaled,
    y,
    y_classes,
    num_classes,
    num_anomalies,
    percentage_keep,
    prec_model=None,
    model_name="DTECategorical",
):
    logger = logging.getLogger("iter()")

    if prec_model is None:
        prec_model = select_model(model_name)

    # Model prediction on current iteration
    logger.info("Train")
    model, new_train_loss = prec_model.fit(X_train, give_train_losses=True)

    logger.info("inferences")
    t_pred = model.predict_score(X_scaled)

    scores = np.array(t_pred)
    y_pred = pred_from_scores(scores, num_anomalies)

    # Calculate the F1
    F1 = skm.f1_score(y, y_pred)

    # getting the most trustable data for next iteration (the ones with lowest scores)
    normal_indices = get_normal_indices(scores, percentage_keep)

    # if we have multi-class anomalies (for exemple with synthetic data we have different types of anomalies, the code
    # just bellow analysis which types of anomalies are still present in the training set, they are the ones that we
    # struggle to eliminate from the training subset). In most case when we don't want to do explainability the code
    # bellow just made no sens and the values will be None
    ano_in_training_set = count_ano(normal_indices, y) / num_anomalies
    count_ano_per_classes = np.zeros(num_classes - 1)
    for idx in normal_indices:
        if y_classes[idx] != 0:
            if num_classes == 2:
                count_ano_per_classes[0] += 1
            elif y_classes[idx] - 1 >= len(count_ano_per_classes):
                count_ano_per_classes[-1] += 1
            else:
                count_ano_per_classes[y_classes[idx] - 1] += 1
    c = count_ano(normal_indices, y)
    if c != 0:
        count_ano_per_classes = count_ano_per_classes / c
    else:
        count_ano_per_classes = None

    logger.info(f"count_ano_per_classes : {count_ano_per_classes}")

    X_train = X_scaled[normal_indices, :]

    return ano_in_training_set, F1, new_train_loss, model, X_train


def itered(
    num_iteration, X, y, y_classes, num_classes, dataset_name, percentage_keep
):
    """
    Takes:
    |
    Return:
    |
    Description
    |
    """
    logger = logging.getLogger("itered()")
    # Données à récupérer au cours des itérations
    data = {
        "nb_anos_in_training": [1],
        "F1_scores": [],
        "train_losses": [],
        "models": [],
    }

    # Traitement des données :
    indices = np.arange(len(X))
    anomaly_indices = indices[y == 1]
    num_anomalies = len(anomaly_indices)
    X_scale = X[indices, :]
    scaler = StandardScaler().fit(X_scale)
    X_scaled = scaler.transform(X_scale)

    # First iteration is unsupervised so X_train = X_scaled

    logger.info("---------------iteration : " + str(0) + " ---------------")
    nb_ano_in_training, F1_score, train_loss, model, X_train = iter(
        X_scaled,
        X_scaled,
        y,
        y_classes,
        num_classes,
        num_anomalies,
        percentage_keep,
    )
    data["nb_anos_in_training"].append(nb_ano_in_training)
    data["F1_scores"].append(F1_score)
    # data['train_losses'].extend(train_loss)
    data["models"].append(model)

    logger.info(f"F1 : {F1_score}")

    for i in range(num_iteration):
        logger.info(f"---------------iteration : {i+1} ---------------")
        nb_ano_in_training, F1_score, train_loss, model, X_train = iter(
            X_train,
            X_scaled,
            y,
            y_classes,
            num_classes,
            num_anomalies,
            percentage_keep,
        )
        data["nb_anos_in_training"].append(nb_ano_in_training)
        data["F1_scores"].append(F1_score)
        data["train_losses"].extend(train_loss)
        data["models"].append(model)
        logger.info(f"F1 : {F1_score}")

    data["nb_anos_in_training"].pop()

    unsup_F1 = data["F1_scores"][0]
    itered_F1 = data["F1_scores"][-1]
    plot_f1_and_loss(
        data=data,
        dataset_name=dataset_name,
        X=X,
        num_anomalies=num_anomalies,
        percentage_keep=percentage_keep,
        num_iteration=num_iteration,
    )
    return model, unsup_F1, itered_F1


# ____________________________________________________________________________
#                                                                           |
#    IV) Explainability with selective noise and Shapley values             |
# ___________________________________________________________________________|


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


def main(args):
    configure_logger()
    logger = logging.getLogger("main()")
    dataSet = args.dataSet

    dataset_name = os.path.splitext(os.path.basename(dataSet))[0]
    extension = os.path.splitext(os.path.basename(dataSet))[-1]
    logger.info("\n\n\n")
    logger.info("====================================")

    # read dataset
    X, y = None, None
    if extension == ".csv":
        df = pd.read_csv(dataSet, header=None)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    elif extension == ".npz":
        data = np.load(dataSet, allow_pickle=True)
        X, y = data["X"], data["y"]

    if not os.path.exists("./res/" + dataset_name):
        os.makedirs("./res/" + dataset_name)

    y_bin, num_classes = map_anomalies(y)
    logger.info(
        f"Dataset: {dataset_name} | Dimensions : ({np.array(X).shape[0]}, {np.array(X).shape[1]})"
    )

    semi_F1 = semi_supervised(
        X, y_bin, dataset_name, model_name="DTEInverseGamma"
    )

    if args.perc_keep:

        # We apply the model for percentage_keep varying from 0.3 to 0.9 and save the results by percentage in
        # resultats_keep_percentage.txt

        _, unsup_F1_1, itered_F1_1 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.3
        )
        _, unsup_F1_2, itered_F1_2 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.4
        )
        model, unsup_F1_3, itered_F1_3 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.5
        )
        _, unsup_F1_4, itered_F1_4 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.6
        )
        _, unsup_F1_5, itered_F1_5 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.7
        )
        _, unsup_F1_6, itered_F1_6 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.8
        )
        _, unsup_F1_7, itered_F1_7 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.9
        )

        unsup_F1 = np.mean(
            [
                unsup_F1_1,
                unsup_F1_2,
                unsup_F1_3,
                unsup_F1_4,
                unsup_F1_5,
                unsup_F1_6,
                unsup_F1_7,
            ]
        )
        result_file = "./resultats_keep_percentage.txt"
        with open(result_file, "a") as f:
            f.write(
                f"{dataset_name} | {round(semi_F1,3)} | {round(unsup_F1,3)} | {round(itered_F1_1,3)} | \
                {round (itered_F1_2,3)} | {round(itered_F1_3,3)} | {round(itered_F1_4,3)} | \
                {round(itered_F1_5,3)} | {round(itered_F1_6,3)} | {round(itered_F1_7,3)}\n"
            )
    else:

        # We apply the model for percentage_keep=0.5 and save the results in resultats_simulation.txt

        model, unsup_F1, itered_F1 = itered(
            10, X, y_bin, y, num_classes, dataset_name, percentage_keep=0.5
        )
        result_file = "./res/resultats_simulation.csv"
        with open(result_file, "a") as f:
            f.write(
                f"{dataset_name} | {round(semi_F1,3)} | {round(unsup_F1,3)} | {round(itered_F1,3)}\n"
            )

    if args.selective_noise:

        # We apply selective noise and retrieve the scores by features, we save these scores in
        # selective_noise_scores.txt, the selective_noise.png graph represents these scores in a bar plot, and
        # couple_noise represents the scores by adding noise to pairs (with a heatmap)

        scores, couple_score = selective_noise(model, X)
        barplot(
            scores,
            xlabel="différentes features",
            ylabel="Scores",
            title="Scores des différentes features sur " + dataset_name,
            file_path="res/" + dataset_name + "/selective_noise.png",
        )

        # Matrix of scores obtained by applying noise on couple of feature
        plt.figure()
        plt.imshow(couple_score, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title("Scores par bruitage sélectif de couples de variables")
        plt.savefig("res/" + dataset_name + "/couple_noise.png")

        with open(
            "res/" + dataset_name + "/selective_noise_scores.txt", "w"
        ) as f:
            for score in scores:
                f.write(f"{score}\n")

    if args.shapley_values:

        # We calculate the importance of the different features with Shapley values, these scores by features are saved
        # in shap_scores.csv and visualizable in shap_values.png

        shap_values = shap_importances_per_feature(model, X)
        importances = np.zeros(shap_values.shape[-1])

        for i in range(len(shap_values)):
            for j in range(len(shap_values[0])):
                importances[j] += np.abs(shap_values[i][j].values)
        barplot(
            importances,
            xlabel="différentes features",
            ylabel="Importances",
            title="Scores des features avec shapley values sur " + dataset_name,
            file_path="res/" + dataset_name + "/shap_values.png",
        )

        with open("res/" + dataset_name + "/shap_scores.txt", "w") as f:
            for score in importances:
                f.write(f"{score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="dte_itered",
        description="This program perform iteration on dte model and provide a \
            new explainability methods comparable with shapley values",
    )
    parser.add_argument("-d", "--dataSet", help="data file")
    parser.add_argument(
        "-n",
        "--selective_noise",
        help="calculate the selective noise",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--shapley_values",
        help="calculate the shapley values",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--perc_keep",
        help="Calculate F1 for different percentage_keep values",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
