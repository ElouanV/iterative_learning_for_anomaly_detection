import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def barplot(
    scores, exp_name, xlabel="", ylabel="", title="", file_path="res/barplot"
):
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

    plt.savefig(file_path)
    plt.show()


def plot_f1_and_loss(
    data,
    dataset_name,
    X,
    exp_name,
    num_anomalies,
    percentage_keep,
    num_iteration,
):

    # Create a figure with 2 subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # First subplot: F1 scores
    axs[0].plot(data["F1_scores"], label="F1_scores")
    axs[0].plot(
        data["nb_anos_in_training"], label="nb_anos_in_training", color="green"
    )
    axs[0].set_xlabel("itération")
    axs[0].set_ylabel("score")
    axs[0].set_title(f"Evolution scores, {exp_name}")
    axs[0].grid()
    axs[0].legend()

    # Second subplot: Training Loss
    n_point_loss = num_iteration + 1
    indices_plot = np.linspace(
        0, len(data["train_losses"]) - 1, n_point_loss
    ).astype(int)
    train_losses_sampled = [data["train_losses"][i] for i in indices_plot]
    axs[1].plot(
        np.linspace(0, len(data["F1_scores"]) - 1, n_point_loss),
        train_losses_sampled,
        label="Train Loss",
    )
    axs[1].set_xlabel("itération")
    axs[1].set_title("Evolution Loss")
    axs[1].grid()
    axs[1].legend()

    fig.suptitle(
        f"{dataset_name}\nNombre d'observations: {X.shape[0]}\n Nombre de features: {X.shape[1]}\n \
            proportion d'anomalies: {(num_anomalies*100/X.shape[0]):.2f}%",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, bottom=0.2)
    plt.text(
        0.5,
        0.01,
        f"Métriques additionnelles:\n E(F1) =  {np.mean(data['F1_scores']):.2f}\n",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
        fontsize=12,
    )

    plt.savefig(f"./res/{dataset_name}/summary_p_{str(percentage_keep)}.png")
    plt.show()



def plot_anomaly_score_distribution(
    scores, saving_path, exp_name, threshold=None
):
    plt.figure(figsize=(10, 10))
    sns.kdeplot(scores, label="Anomaly score distribution")
    if threshold is not None:
        plt.axvline(threshold, color="r", linestyle="--", label="Threshold")
    plt.legend()
    plt.grid()
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(f"Anomaly score distribution, {exp_name}")
    plt.savefig(Path(saving_path, "anomaly_score_distribution.png"))
    plt.show()


def plot_anomaly_score_distribution_split(
    scores, y_test_anomaly, exp_name, p, saving_path
):
    plt.figure(figsize=(10, 10))
    sns.kdeplot(scores[y_test_anomaly == 0], label="Normal score distribution")
    sns.kdeplot(scores[y_test_anomaly == 1], label="Anomaly score distribution")
    sns.kdeplot(scores[p == 1], label="Detected anomaly score distribution")
    plt.legend()
    plt.grid()
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(f"Anomaly score distribution, {exp_name}")
    plt.savefig(Path(saving_path, "anomaly_score_distribution_split.png"))
    plt.show()
    plt.clf()


def plot_score_density(scores, exp_name, saving_path):
    plt.figure(figsize=(10, 10))
    sns.kdeplot(scores, label="Anomaly score distribution")
    plt.grid()
    plt.legend()
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(f"Anomaly score distribution, {exp_name}")
    plt.savefig(Path(saving_path, "score_density.png"))
    plt.show()


def plot_feature_importance(
    scores,
    exp_name,
    expected_explanation=None,
    feature_names=None,
    xlabel="",
    ylabel="",
    saving_path="",
):
    """
    Takes:
    |   scores : the scores to plot (should be a score per feature)
    |   feature_names : the name of the features
    |   other args are explicit and used for ploting
    Description :
    |   plot the scores of the different feature on a barplot and saves it in file_path
    """
    plt.figure(figsize=(10, 6))
    # Normalize the scores
    scores = np.abs(scores)
    scores = scores / np.sum(scores)
    plt.bar(range(0, len(scores)), scores.squeeze(), label="Feature importance")
    if expected_explanation is not None:
        plt.bar(
            range(0, len(scores)),
            expected_explanation.squeeze(),
            alpha=0.5,
            label="Expected explanation",
        )
    if feature_names is not None:
        plt.xticks(range(1, len(scores) + 1), feature_names, rotation=90)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f"Feature importance, {exp_name}")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(saving_path, f"feature_importance_{exp_name}.png"))
    plt.show()
    plt.close()


def plot_couple_feature_importance_matrix(
    scores,
    exp_name,
    feature_names="auto",
    xlabel="",
    ylabel="",
    saving_path="",
):
    """
    Takes:
    |   scores : the scores to plot (should be a score per feature)
    |   feature_names : the name of the features
    |   other args are explicit and used for ploting
    Description :
    |   plot the scores of the different feature on a barplot and saves it in file_path
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        scores, annot=True, xticklabels=feature_names, yticklabels=feature_names
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Couple feature importance matrix, " + exp_name)
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(saving_path, "couple_feature_importance_matrix.png")
    )
    plt.show()


def plot_nmf(nmf_w, nmf_h, exp_name, saving_path):
    # Plot NMF results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(
        nmf_w,
        cmap="viridis",
    )
    plt.title("NMF W matrix")
    plt.xlabel("Patterns")
    plt.ylabel("Anomaly")
    plt.subplot(1, 2, 2)
    sns.heatmap(nmf_h, cmap="viridis", xticklabels=range(1, nmf_h.shape[1] + 1))
    plt.title(f"NMF H matrix, {exp_name}")
    plt.xlabel("Features")
    plt.ylabel("Patterns")
    plt.savefig(os.path.join(saving_path, "nmf.png"))
    plt.show()


def reconstruction_error_boxplot(reconstruction_error, exp_name, saving_path):
    # Box plot of the error
    plt.figure(figsize=(10, 6))
    plt.boxplot(reconstruction_error)
    plt.xlabel("Feature")
    plt.ylabel("Mean Error")
    plt.title(
        f"Box plot of the reconstruction error for each feature, {exp_name}"
    )
    plt.grid()
    plt.savefig(os.path.join(saving_path, "reconstruction_error_boxplot.png"))
    plt.show()


def reconstruction_error_plot(
    anomaly_reconstruction, normal_reconstruction, exp_name, saving_path
):
    plt.figure(figsize=(10, 6))
    anomaly_mean_error = np.mean(anomaly_reconstruction, axis=0)
    normal_mean_error = np.mean(normal_reconstruction, axis=0)
    print(anomaly_mean_error.shape)
    plt.bar(
        range(anomaly_mean_error.shape[0]),
        anomaly_mean_error,
        label="Anomaly",
        color="orange",
        alpha=0.3,
    )
    plt.bar(
        range(normal_mean_error.shape[0]),
        normal_mean_error,
        label="Normal",
        alpha=0.3,
        color="green",
    )
    plt.xlabel("Feature")
    plt.legend()
    plt.ylabel("Mean Error")
    plt.title(
        f"Mean Absolute reconstruction error for each feature, {exp_name}"
    )
    plt.grid()
    plt.savefig(os.path.join(saving_path, "reconstruction_error.png"))
    plt.show()


def iterative_training_score_evolution(iteration_scores, exp_name, saving_path):
    plt.figure(figsize=(10, 6))

    for i, scores in enumerate(iteration_scores):
        sns.kdeplot(scores, label=f"Iteration {i}")
    plt.legend()
    plt.grid()
    plt.title(f"Anomaly score evolution during iterative training, {exp_name}")
    plt.savefig(
        os.path.join(saving_path, "iterative_training_score_evolution.png")
    )
    plt.show()


def plot_tsne(tsne_model, X, y, iteration, saving_path, indices_to_keep):
    X_embedded = tsne_model.fit_transform(X)
    plt.figure(figsize=(10, 10))
    keep_mask = np.full(X.shape[0], 0)
    keep_mask[indices_to_keep] = 1
    for anomaly_type in np.unique(y):
        mask = y == anomaly_type
        mask_keep = mask & keep_mask
        plt.scatter(
            X_embedded[mask_keep, 0],
            X_embedded[mask_keep, 1],
            label=f"Anomaly type {anomaly_type}",
            alpha=1,
        )
        mask_not_keep = mask & ~keep_mask
        plt.scatter(
            X_embedded[mask_not_keep, 0],
            X_embedded[mask_not_keep, 1],
            label=f"Anomaly type {anomaly_type}",
            alpha=0.2,
        )
    plt.legend()
    plt.title(f"t-SNE visualization at iteration {iteration}")
    plt.grid()
    plt.savefig(os.path.join(saving_path, f"tsne_iteration_{iteration}.png"))
    plt.show()