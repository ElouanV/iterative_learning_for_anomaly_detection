from sklearn.linear_model import Ridge
import numpy as np


class CustomLIME:
    def __init__(self, model_predict_fn, kernel_width, surrogate_model):
        self.model_predict_fn = model_predict_fn
        self.kernel_width = kernel_width
        self.surrogate_model = surrogate_model or Ridge(alpha=1.0)

    def kernel_fn(self, distances):
        return np.sqrt(np.exp(-distances / (2 * self.kernel_width**2)))

    def explain_instance(self, x, diffusion_perturb_fn, num_samples=1000):
        # Generate perturbed samples
        perturbed_samples = diffusion_perturb_fn(x, num_samples)

        # Get predictions for perturbed samples
        predictions = self.model_predict_fn(perturbed_samples)

        # Compute distances from the original instance
        distances = np.linalg.norm(perturbed_samples - x, axis=1)

        # Compute kernel weights
        weights = self.kernel_fn(distances)

        # Fit the surrogate model
        self.surrogate_model.fit(
            perturbed_samples, predictions, sample_weight=weights
        )

        # Get feature importances
        feature_importances = self.surrogate_model.coef_

        return {
            "feature_importances": feature_importances,
            "intercept": self.surrogate_model.intercept_,
            "perturbed_samples": perturbed_samples,
            "predictions": predictions,
            "weights": weights,
        }
