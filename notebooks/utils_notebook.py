import matplotlib.pyplot as plt
import numpy as np


def histogram_experiment(
    mean_df, std_df, column, title, ylabel, synthetic=True
):
    if synthetic:
        mean_df = mean_df.sort_values(
            by=["dimension", "anomaly_ratio", "dataset_name"]
        )
        std_df = std_df.sort_values(
            by=["dimension", "anomaly_ratio", "dataset_name"]
        )
    datasets = mean_df["dataset_name"].unique()
    experiments = mean_df["experiment"].unique()
    bar_width = 0.2
    num_datasets = len(datasets)
    num_experiments = len(experiments)
    r = np.arange(num_datasets)
    _, ax = plt.subplots(figsize=(10, 6))
    for i, experiment in enumerate(experiments):
        mean_experiment = mean_df[mean_df["experiment"] == experiment]
        std_experiment = std_df[std_df["experiment"] == experiment]
        mean_experiment = mean_experiment.sort_values(by="dataset_name")[
            column
        ].values
        std_experiment = std_experiment.sort_values(by="dataset_name")[
            column
        ].values
        ax.bar(
            r + i * bar_width,
            mean_experiment,
            width=bar_width,
            yerr=std_experiment,
            label=experiment,
        )
    ax.set_xlabel("Datasets")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(r + bar_width * (num_experiments) / 2)
    ax.set_xticklabels(datasets)

    ax.legend()
    plt.grid()
    plt.xticks(rotation=90)
    plt.show()


def dataframe_to_latex(
    df,
    column_format=None,
    caption=None,
    label=None,
    header=True,
    index=False,
    float_format="%.2f",
):
    if column_format is None:
        column_format = "l" * (len(df.columns) + (1 if index else 0))
    # For float columns, round to 2 decimal places
    for col in df.select_dtypes(include=[np.float64, np.float32]).columns:
        df[col] = df[col].round(2)
    # Convert the DataFrame to LaTeX
    latex_str = df.to_latex(
        index=index,
        column_format=column_format,
        header=header,
        float_format=float_format,
        escape=False,  # Allows LaTeX-specific characters (e.g., \%)
    )

    # Add optional caption and label
    if caption or label:
        latex_table = "\\begin{table}[ht]\n\\centering\n"
        latex_table += latex_str
        if caption:
            latex_table += f"\\caption{{{caption}}}\n"
        if label:
            latex_table += f"\\label{{{label}}}\n"
        latex_table += "\\end{table}"
    else:
        latex_table = latex_str

    return latex_table
