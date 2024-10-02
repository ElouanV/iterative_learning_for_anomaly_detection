import matplotlib as plt


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
