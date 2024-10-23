import torch
from sklearn.decomposition import NMF
from torch.utils.data import DataLoader
from tqdm import tqdm


class DDPMExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def feature_wise_explanation(
        self, X_eval, y_eval, saving_path, bath_size=32, n_components=2
    ):

        # For each data of the eval set, compute reconstruction error
        X_anomaly = X_eval[y_eval == 1]
        anomaly_eval_data_loader = DataLoader(
            X_anomaly,
            batch_size=bath_size,
            shuffle=False,
        )
        X_normal = X_eval[y_eval == 0]
        normal_eval_data_loader = DataLoader(
            X_normal,
            batch_size=bath_size,
            shuffle=False,
        )
        anomaly_reconstruction_errors = []
        for _, (X) in tqdm(enumerate(anomaly_eval_data_loader), desc="Anomaly"):
            X = X.to(self.device)
            reconstructed_x = self.model.raw_prediction(X)[-1]
            reconstructed_x = reconstructed_x.to("cpu")
            X = X.to("cpu")
            # Compute reconstruction error
            error = torch.abs(X - reconstructed_x)
            anomaly_reconstruction_errors.append(error)
        normal_reconstruction_errors = []
        for _, (X) in tqdm(enumerate(normal_eval_data_loader), desc="Normal"):
            X = X.to(self.device)
            reconstructed_x = self.model.raw_prediction(X)[-1]
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
