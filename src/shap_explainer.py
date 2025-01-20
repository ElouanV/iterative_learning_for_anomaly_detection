import logging
import os

import numpy as np
import shap
import torch

from src.viz.training_viz import plot_feature_importance


class ShapExplainer:
    def __init__(self, model, data):
        self.model = model

        def f(x):
            return self.model.predict_score(x)

        data = shap.sample(data, 100)
        self.explainer = shap.KernelExplainer(f, data)
        self.logger = logging.getLogger(__name__)

    def explain_instance(self, x):
        explanations = []
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        self.logger.info("Shap explainer running")
        shap_values = self.explainer.shap_values(
            x, check_additivity=False, nsamples="auto", silent=True
        )
        self.logger.info("Shap explainer done")
        explanations.append(shap_values)

        explanations = np.array(explanations)
        return explanations
