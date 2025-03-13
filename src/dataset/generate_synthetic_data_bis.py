import os
import pickle
import random

import hydra
import networkx as nx
import numpy as np
import omegaconf
import yaml
# Import bidict
from bidict import bidict
from matplotlib import pyplot as plt

from utils import split_list


def generate_random_dag_with_edges(num_nodes: int, num_edges: int):
    # Ensure the number of edges does not exceed the maximum possible acyclic edges
    max_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_edges:
        raise ValueError(
            "Too many edges requested for the number of nodes. Maximum possible edges for DAG exceeded."
        )

    # Create an empty directed graph
    dag = nx.DiGraph()

    # Add nodes to the graph
    dag.add_nodes_from(range(num_nodes))

    # Generate a topological order of nodes
    nodes = list(dag.nodes)
    random.shuffle(nodes)

    # Generate all possible edges (u, v) where u appears before v in topological order
    possible_edges = [
        (nodes[i], nodes[j])
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
    ]

    # Randomly select the required number of edges
    selected_edges = random.sample(possible_edges, num_edges)

    # Add edges to the graph
    dag.add_edges_from(selected_edges)

    return dag


def run_graphical_model(
    graphical_model,
    number_of_samples: int,
):
    """ """
    X = np.zeros((number_of_samples, len(graphical_model.nodes)))
    for node in graphical_model.nodes:
        X[:, node] = graphical_model.nodes[node]["fn"](
            *graphical_model.nodes[node]["distrib_params"], number_of_samples
        )
    for src, dst in graphical_model.edges:
        X[:, dst] += X[:, src]
    return X


def plot_graph(graphical_model, saving_path=None):
    """ """
    pos = nx.spring_layout(graphical_model)
    nx.draw(graphical_model, pos, with_labels=True, arrows=True)
    plt.title("Graphical Model")
    plt.savefig(saving_path + "/graphical_model.png")
    plt.show()


def select_feature(n_features, n):
    """ """
    selected_features = np.random.choice(np.array(range(n_features)), n)
    return selected_features


def add_anomalies(
    X, graphical_model, anomalies_scheme: dict, ratio_anomalies: list
):
    assert len(anomalies_scheme.keys()) == len(ratio_anomalies)
    splits = split_list(list(range(X.shape[0])), ratio_anomalies)
    explanation = np.zeros(X.shape)
    y = np.zeros(X.shape[0])
    # Use bidict to store the type of anomaly and an int
    anomaly_code = {
        "cluster": 1,
        "global": 2,
        "additive_noise": 4,
        "multiplicative_noise": 5,
        "local": 6,
    }
    anomaly_code = bidict(anomaly_code)
    for anomaly, indices in zip(anomalies_scheme.keys(), splits):
        anomalies_info = anomalies_scheme[anomaly]
        if anomaly == "cluster":
            alpha = anomalies_info["alpha"]
            modified_graphical_model = graphical_model.copy()
            for node in modified_graphical_model.nodes:
                modified_graphical_model.nodes[node]["distrib_params"] = [
                    alpha * param
                    for param in modified_graphical_model.nodes[node][
                        "distrib_params"
                    ]
                ]
            X[indices] = run_graphical_model(
                modified_graphical_model, len(indices)
            )
            y[indices] = anomaly_code[anomaly]
            explanation[indices] = np.ones(X[indices].shape)
        elif anomaly == "global":
            indices = np.array(indices).reshape(-1, 1)
            selected_features = anomalies_info["features"]
            alpha = anomalies_info["alpha"]
            low = alpha * X[:, selected_features].min(axis=0)
            high = X[:, selected_features].max(axis=0) * alpha
            new_values = np.random.uniform(
                low, high, X[indices, selected_features].shape
            )
            y[indices] = anomaly_code[anomaly]
            explanation[indices] = np.zeros(X[indices].shape)
            explanation[indices, selected_features] = 1
            X[indices, selected_features] = new_values
        elif anomaly == "local":
            selected_features = anomalies_info["features"]
            indices = np.array(indices).reshape(-1, 1)
            # Get covariance matrix
            cov = np.cov(X[:, selected_features].T)
            # scale the covariance matrix by alpha
            alpha = anomalies_info["alpha"]
            cov *= alpha
            # Generate new samples
            X[indices, selected_features] = np.random.multivariate_normal(
                X[:, selected_features].mean(axis=0), cov, len(indices)
            )
            y[indices] = anomaly_code[anomaly]
            explanation[indices, selected_features] = 1
        elif anomaly == "additive_noise":
            selected_features = anomalies_info["features"]
            indices = np.array(indices).reshape(-1, 1)
            alpha = anomalies_info["alpha"]
            noise = np.random.normal(
                X[:, selected_features].std() * alpha,
                X[:, selected_features].std(),
                X[indices, selected_features].shape,
            )
            X[indices, selected_features] += noise
            y[indices] = anomaly_code[anomaly]
            explanation[indices, selected_features] = 1
        elif anomaly == "multiplicative_noise":
            indices = np.array(indices).reshape(-1, 1)
            selected_features = anomalies_info["features"]
            alpha = anomalies_info["alpha"]
            noise = np.random.normal(
                X[:, selected_features].std() * alpha,
                0.3,
                X[indices, selected_features].shape,
            )
            X[indices, selected_features] *= noise
            y[indices] = anomaly_code[anomaly]
            explanation[indices, selected_features] = 1
        else:
            raise ValueError("Anomaly type not recognized")
    return X, y, explanation


def generate_graphical_model(
    n_features: int,
    number_of_causalities: int,
    distributions: list,
    saving_path: str,
):
    feature_distributions = random.choices(distributions, k=n_features)
    graphical_model = generate_random_dag_with_edges(
        n_features, number_of_causalities
    )
    # generate the samples according to the graphical model
    feature_generation_info = {}
    for i, distrib in enumerate(feature_distributions):
        if distrib == "uniform":
            low = np.random.uniform(0, 3)
            high = low + np.random.uniform(1, 10)
            fn = np.random.uniform
            distrib_params = [low, high]
            # save the function in the node attribute of the graphical model
            feature_generation_info[i] = ("uniform", [low, high])
        elif distrib == "normal":
            loc = np.random.uniform(-1, 5)
            scale = np.random.uniform(1, 7)
            fn = np.random.normal
            distrib_params = [loc, scale]

            feature_generation_info[i] = ("normal", [loc, scale])
        elif distrib == "exponential":
            scale = np.random.uniform(1, 5)
            fn = np.random.exponential
            distrib_params = [scale]
            feature_generation_info[i] = ("exponential", [scale])
        elif distrib == "gamma":
            shape = np.random.uniform(1, 5)
            scale = np.random.uniform(1, 5)
            fn = np.random.gamma
            distrib_params = [shape, scale]
            feature_generation_info[i] = ("gamma", [shape, scale])
        graphical_model.nodes[i]["fn"] = fn
        graphical_model.nodes[i]["distrib_params"] = distrib_params
    return graphical_model, feature_generation_info


def generate_synthetic_data(
    n_samples: int,
    ratio_aomalies: list,
    n_features: int,
    n_causalities: int,
    anomalies: list,
    distributions: list,
    saving_path: str,
    db_name: str,
):
    """ """
    os.makedirs(saving_path, exist_ok=True)
    graphical_model = generate_graphical_model(
        n_features,
        n_causalities,
        distributions=distributions,
        saving_path=saving_path,
    )
    X = run_graphical_model(graphical_model, n_samples)
    plot_graph(graphical_model, saving_path="test")
    X, y, explanation, anomaly_information = add_anomalies(
        X, graphical_model, anomalies=anomalies, ratio_anomalies=ratio_aomalies
    )

    #######
    data = {
        "X": np.array(X),
        "y": np.array(y),
        "explanation": np.array(explanation),
    }
    with open(
        os.path.join(saving_path, "anomaly_information.yaml"), "w"
    ) as file:
        yaml.dump(anomaly_information, file)
    pickle.dump(data, open(os.path.join(saving_path, "data.npy"), "wb"))
    np.save(os.path.join(saving_path, "features.npy"), np.array(X))
    np.save(os.path.join(saving_path, "labels.npy"), np.array(y))
    np.save(os.path.join(saving_path, "explanation.npy"), np.array(explanation))
    # Generate yalm config file:
    config = {
        "data_type": "tabular",
        "dataset_name": db_name,
        "dataset_path": os.path.join(saving_path, "data.npy"),
        "test_ratio": 0.2,
    }
    config_path = "conf/dataset"
    with open(os.path.join(config_path, f"{db_name}.yaml"), "w") as file:
        yaml.dump(config, file)


def generate_anomalies_scheme(anomalies: list, n_features):
    # Use bidict to store the type of anomaly and an int
    anomaly_code = {
        "cluster": 1,
        "global": 2,
        "dependecy": 3,
        "additive_noise": 4,
        "multiplicative_noise": 5,
        "local": 6,
    }
    anomaly_code = bidict(anomaly_code)
    anomaly_scheme = {}
    for anomaly in anomalies:
        if anomaly == "cluster":
            alpha = 3
            anomaly_scheme[anomaly] = {"alpha": alpha, "features": "all"}
        elif anomaly == "global":
            high = int(n_features / 2)
            if high <= 1:
                high = 2
            n = np.random.randint(1, high)
            selected_features = select_feature(n_features, n)
            alpha = 1.025
            anomaly_scheme[anomaly] = {
                "alpha": alpha,
                "features": selected_features.tolist(),
            }
        elif anomaly == "dependecy":
            # Remove the causality between the features
            pass
        elif anomaly == "local":
            high = int(n_features / 2)
            if high <= 2:
                n = 3
            else:
                n = np.random.randint(2, high)
            selected_features = select_feature(n_features=n_features, n=n)
            # scale the covariance matrix by alpha
            alpha = 1.05
            anomaly_scheme[anomaly] = {
                "alpha": alpha,
                "features": selected_features.tolist(),
            }
        elif anomaly == "additive_noise":
            # Add a random noise to a feature
            high = int(n_features / 2)
            if high <= 1:
                high = 2
            n = np.random.randint(1, high)
            selected_features = select_feature(n_features, n)
            alpha = 3
            anomaly_scheme[anomaly] = {
                "features": selected_features.tolist(),
                "alpha": alpha,
            }
        elif anomaly == "multiplicative_noise":
            high = int(n_features / 2)
            if high <= 1:
                high = 2
            n = np.random.randint(1, high)
            selected_features = select_feature(n_features, n)
            alpha = 0.5

            anomaly_scheme[anomaly] = {
                "features": selected_features.tolist(),
                "alpha": alpha,
            }
        else:
            raise ValueError("Anomaly type not recognized")
    return anomaly_scheme


def generate_dataset(
    graphical_model, anomalies_scheme, n_samples, ratio_anomalies, seed=42
):
    # Set the seed
    np.random.seed(seed)
    random.seed(seed)

    X = run_graphical_model(graphical_model, n_samples)
    X, y, explanation = add_anomalies(
        X, graphical_model, anomalies_scheme, ratio_anomalies
    )
    return X, y, explanation


def generate_data_wip(
    n_samples: int,
    ratio_anomalies: list,
    n_features: int,
    n_causalities: int,
    anomalies: list,
    distributions: list,
    saving_path: str,
    db_name: str,
):

    # Generate graphical model
    graphical_model, feature_generation_information = generate_graphical_model(
        n_features,
        n_causalities,
        distributions=distributions,
        saving_path=saving_path,
    )
    anomalies_scheme = generate_anomalies_scheme(
        anomalies=anomalies, n_features=n_features
    )
    # Generate the data
    for seed in [0, 1, 2, 3, 4, 5]:
        version_db_name = db_name + f"_seed_{seed}"
        version_saving_path = os.path.join(saving_path, version_db_name)
        os.makedirs(version_saving_path, exist_ok=True)
        plot_graph(graphical_model, saving_path=version_saving_path)
        with open(
            os.path.join(version_saving_path, "anomalies_scheme.yaml"), "w"
        ) as file:
            yaml.dump(anomalies_scheme, file)
            # Save the feature generation info in a yaml file
        with open(
            os.path.join(saving_path, "feature_generation_info.yaml"), "w"
        ) as file:
            yaml.dump(feature_generation_information, file)
        X, y, explanation = generate_dataset(
            graphical_model,
            anomalies_scheme,
            n_samples,
            ratio_anomalies,
            seed=seed,
        )
        data = {
            "X": np.array(X),
            "y": np.array(y),
            "explanation": np.array(explanation),
        }
        with open(
            os.path.join(version_saving_path, "anomaly_information.yaml"), "w"
        ) as file:
            yaml.dump(anomalies_scheme, file)
        pickle.dump(
            data, open(os.path.join(version_saving_path, "data.npy"), "wb")
        )
        np.save(os.path.join(version_saving_path, "features.npy"), np.array(X))
        np.save(os.path.join(version_saving_path, "labels.npy"), np.array(y))
        np.save(
            os.path.join(version_saving_path, "explanation.npy"),
            np.array(explanation),
        )
        # Generate yalm config file:
        config = {
            "data_type": "tabular",
            "dataset_name": version_db_name,
            "dataset_path": os.path.join(version_saving_path, "data.npy"),
            "test_ratio": 0.2,
        }
        config_path = "conf/dataset"
        with open(
            os.path.join(config_path, f"{version_db_name}.yaml"), "w"
        ) as file:
            yaml.dump(config, file)


@hydra.main(
    version_base=None, config_path="../../conf", config_name="config_datagen"
)
def main(cfg: omegaconf.DictConfig):
    anomalies_ratio_str = "_".join(map(str, cfg.dataset.ratio_anomalies))
    db_name = (
        f"A_synthetic_f{cfg.dataset.n_features}_s{cfg.dataset.n_samples}_"
        f"c{cfg.dataset.n_causalities}_r{anomalies_ratio_str}"
    )
    os.makedirs(os.path.join(cfg.dataset.saving_path, db_name), exist_ok=True)
    generate_data_wip(
        n_samples=cfg.dataset.n_samples,
        ratio_anomalies=cfg.dataset.ratio_anomalies,
        n_features=cfg.dataset.n_features,
        n_causalities=cfg.dataset.n_causalities,
        anomalies=cfg.dataset.anomalies,
        distributions=cfg.dataset.distributions,
        saving_path=cfg.dataset.saving_path,
        db_name=db_name,
    )


if __name__ == "__main__":
    main()
