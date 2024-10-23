import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    plt.grid()

    plt.savefig(file_path)
    plt.show()


def plot_f1_and_loss(
    data, dataset_name, X, num_anomalies, percentage_keep, num_iteration
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
    axs[0].set_title("Evolution scores")
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


def plot_tsne(data, y_test_anomaly, p, saving_path):
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(data["X_test"])
    plt.figure(figsize=(10, 10))
    plt.scatter(
        X_embedded[y_test_anomaly == 0, 0],
        X_embedded[y_test_anomaly == 0, 1],
        c="b",
        label="0",
    )
    plt.scatter(
        X_embedded[y_test_anomaly == 1, 0],
        X_embedded[y_test_anomaly == 1, 1],
        c="r",
        label="1",
    )
    indices_of_well_detected_anomalies = np.where(
        (p == 1) & (y_test_anomaly == 1)
    )
    indices_of_badly_detected_anomalies = np.where(
        (p == 1) & (y_test_anomaly == 0)
    )
    plt.scatter(
        X_embedded[indices_of_well_detected_anomalies, 0],
        X_embedded[indices_of_well_detected_anomalies, 1],
        c="g",
        label="detected",
    )
    plt.scatter(
        X_embedded[indices_of_badly_detected_anomalies, 0],
        X_embedded[indices_of_badly_detected_anomalies, 1],
        c="y",
        label="false positive",
    )
    plt.legend()
    plt.savefig(Path(saving_path, "tsne.png"))
    plt.show()


def plot_anomaly_score_distribution(scores, saving_path):
    plt.figure(figsize=(10, 10))
    sns.kdeplot(scores, label="Anomaly score distribution")

    plt.legend()
    plt.grid()
    plt.savefig(Path(saving_path, "anomaly_score_distribution.png"))
    plt.show()


def plot_anomaly_score_distribution_split(
    scores, y_test_anomaly, p, saving_path
):
    plt.figure(figsize=(10, 10))
    sns.kdeplot(scores[y_test_anomaly == 0], label="Normal score distribution")
    sns.kdeplot(scores[y_test_anomaly == 1], label="Anomaly score distribution")
    sns.kdeplot(scores[p == 1], label="Detected anomaly score distribution")
    plt.legend()
    plt.grid()
    plt.savefig(Path(saving_path, "anomaly_score_distribution_split.png"))
    plt.show()
    plt.clf()


def plot_score_density(scores, saving_path):
    plt.figure(figsize=(10, 10))
    sns.kdeplot(scores, label="Anomaly score distribution")
    plt.grid()
    plt.legend()
    plt.savefig(Path(saving_path, "score_density.png"))
    plt.show()


def plot_feature_importance(
    scores, feature_names=None, xlabel="", ylabel="", title="", saving_path=""
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
    plt.bar(range(1, len(scores) + 1), scores, align="center")

    if feature_names is not None:
        plt.xticks(range(1, len(scores) + 1), feature_names, rotation=90)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(saving_path, "feature_importance.png"))
    plt.show()


def plot_couple_feature_importance_matrix(
    scores, feature_names='auto', xlabel="", ylabel="", title="", saving_path=""
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
    plt.title(title)
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(saving_path, "couple_feature_importance_matrix.png"))
    plt.show()


def plot_nmf(nmf_w, nmf_h, saving_path):
    # Plot NMF results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(nmf_w, cmap="viridis")
    plt.title("NMF W matrix")
    plt.xlabel("Patterns")
    plt.ylabel("Transactions")
    plt.subplot(1, 2, 2)
    sns.heatmap(nmf_h, cmap="viridis")
    plt.title("NMF H matrix")
    plt.xlabel("Features")
    plt.ylabel("Patterns")
    plt.savefig(os.path.join(saving_path, "nmf.png"))
    plt.show()


def reconstruction_error_boxplot(reconstruction_error, saving_path):
    # Box plot of the error
    plt.boxplot(reconstruction_error)
    plt.xlabel("Feature")
    plt.ylabel("Mean Error")
    plt.title("Box plot of the reconstruction error for each feature")
    plt.grid()
    plt.savefig(os.path.join(saving_path, "reconstruction_error_boxplot.png"))
    plt.show()
