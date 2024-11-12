import os

import numpy as np
import shap

from viz.training_viz import plot_feature_importance


class ShapExplainer:
    def __init__(self, model, data):
        self.model = model

        def f(x):
            return self.model.predict_score(x)

        data = shap.sample(data, 100)
        self.explainer = shap.KernelExplainer(f, data)

    def explain_instance(
        self, x, expected_explanation, saving_path, experiment_name, plot=True
    ):
        explanations = []
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        for i in range(len(x)):
            shap_values = self.explainer.shap_values(
                x[i], check_additivity=False, nsamples=1000
            )
            explanations.append(shap_values)
            if plot:
                os.makedirs(
                    os.path.join(saving_path, "shap_values"), exist_ok=True
                )
                plot_feature_importance(
                    shap_values,
                    expected_explanation=expected_explanation[i],
                    exp_name=experiment_name + f"shap_value_instance{i}",
                    saving_path=os.path.join(saving_path, "shap_values"),
                )
        explanations = np.array(explanations)
        return explanations
