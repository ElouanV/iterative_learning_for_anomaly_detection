import argparse
import pickle

import numpy as np
import pandas as pd

# → Trois grand types d’altération pour génerer des datasets avec des anomalies:

# 1- altérer les hyperparamètres de certaines variables
# 2- altérer les directions de causalité (enlever ou ajouter des corrélation)
# 3- Ajouter des variables latentes (connecté aux variable observé mais invisible dans le jeu de données)


def generate_synthetic_data(num_samples, add_correlation=False):
    print(add_correlation)
    # Générer les features
    features = {
        "feature1": np.random.normal(loc=0, scale=1, size=num_samples),
        "feature2": np.random.normal(loc=1, scale=2, size=num_samples),
        "feature3": np.random.normal(loc=5, scale=1, size=num_samples),
        "feature4": np.random.normal(loc=0, scale=5, size=num_samples),
    }

    if add_correlation:
        noise = np.random.normal(loc=2, scale=0.5, size=num_samples)
        features["feature5"] = features["feature3"] + noise
    else:
        features["feature5"] = np.random.normal(
            loc=7, scale=1, size=num_samples
        )
    features["temoin1"] = np.random.normal(loc=0, scale=1, size=num_samples)
    features["temoin2"] = np.random.normal(loc=1, scale=2, size=num_samples)
    features["temoin3"] = np.random.normal(loc=2, scale=1, size=num_samples)

    # Créer un DataFrame à partir des features
    df = pd.DataFrame(features)

    # Ajouter une colonne pour le type d'anomalie (0 = pas d'anomalie)
    df["anomaly"] = 0

    return df


def add_anomalies(
    df, prop_anomalies1, prop_anomalies2, prop_anomalies3, add_correlation=False
):
    num_samples = len(df)
    num_anomalies1 = int(prop_anomalies1 * num_samples)
    num_anomalies2 = int(prop_anomalies2 * num_samples)
    num_anomalies3 = int(prop_anomalies3 * num_samples)
    print(num_anomalies1, num_anomalies2, num_anomalies3)

    # 1 outliers : altération des hyperparamètres
    indices = np.random.choice(num_samples, num_anomalies1, replace=False)
    for idx in indices:
        df.at[idx, "feature1"] = np.random.choice([-4, 4])
        df.at[idx, "anomaly"] = 1

    # 2. Altération des directions de causalité (ajout/suppression de corrélation)
    indices = np.random.choice(num_samples, num_anomalies2, replace=False)
    for idx in indices:
        if add_correlation:
            if np.random.rand() > 0.5:
                # Ajouter une corrélation entre feature2 et feature3
                df.at[idx, "feature2"] = df.at[
                    idx, "feature3"
                ] + np.random.normal(loc=0, scale=0.1)
                df.at[idx, "anomaly"] = 2
            else:
                # Enlever la corrélation entre feature5 et feature3
                df.at[idx, "feature5"] = np.random.normal(loc=7, scale=1)
                df.at[idx, "anomaly"] = 3
        else:
            df.at[idx, "feature2"] = df.at[idx, "feature3"] + np.random.normal(
                loc=0, scale=0.1
            )
            df.at[idx, "anomaly"] = 2

    # 3. Ajouter des variables latentes (ajouter des anomalies dans des variables connectées)
    indices = np.random.choice(num_samples, num_anomalies3, replace=False)
    for _ in range(num_anomalies3):
        idx = np.random.randint(0, num_sample)
        latent_effect = np.random.normal(
            loc=5, scale=2
        )  # variable latente non observée
        df.at[idx, "feature3"] += latent_effect
        df.at[idx, "feature4"] += latent_effect
        df.at[idx, "anomaly"] = 4

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--dataSetPath", help="Path to save the generated dataset"
    )
    parser.add_argument(
        "-n",
        "--num_sample",
        type=int,
        default=1000,
        help="Number of row in the dataset",
    )
    parser.add_argument(
        "-a1",
        "--prop_anomalies1",
        type=float,
        default=0.0,
        help="Proportion of anomaly of type 1",
    )
    parser.add_argument(
        "-a2",
        "--prop_anomalies2",
        type=float,
        default=0.0,
        help="Proportion of anomaly of type 2",
    )
    parser.add_argument(
        "-a3",
        "--prop_anomalies3",
        type=float,
        default=0.0,
        help="Proportion of anomaly of type 3",
    )
    parser.add_argument(
        "-c",
        "--add_correlation",
        help="Boolean for having or not a correlation between features 5 and 3",
        action="store_true",
    )
    args = parser.parse_args()
    dataSetPath = args.dataSetPath
    num_sample = args.num_sample
    prop_anomalies1 = args.prop_anomalies1
    prop_anomalies2 = args.prop_anomalies2
    prop_anomalies3 = args.prop_anomalies3
    add_correlation = args.add_correlation

    df = generate_synthetic_data(num_sample, add_correlation)
    df_with_anomalies = add_anomalies(
        df, prop_anomalies1, prop_anomalies2, prop_anomalies3, add_correlation
    )
    data = {}
    X = df_with_anomalies.drop(columns=["anomaly"]).to_numpy()

    y = df_with_anomalies["anomaly"].to_numpy()
    data["X"] = X
    data["y"] = y

    pickle.dump(data, open(dataSetPath, "wb"))
