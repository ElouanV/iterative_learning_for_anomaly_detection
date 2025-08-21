<div align="center">

# Diffusion for Explainable Unsupervised Anomaly Detection
**Iterative Learning for Tabular Anomaly Detection & Explainability**  
<em>(Author list placeholder)</em>

[Paper (arXiv/DOI – coming soon)](./) · (Optional project page)  

</div>

## Overview
This repository contains the reference implementation for the paper "Diffusion for Explainable Unsupervised Anomaly Detection" introducing an iterative sampling / weighting strategy ("DSIL") for unsupervised anomaly detection on tabular data, together with feature-level explanation procedures for diffusion-based time estimation (DTE / DDPM) models.

Core contributions:
- Diffusion Sampling Iterative Learning (DSIL) for progressively refining the training subset using score-based selection schedules (constant, cosine, exponential, exponential_v2; deterministic or probabilistic).
- Weighted-loss iterative refinement implementation from [1].
- Explanation methods: reconstruction error (DDPM), diffusion perturbation (mean / max), gradients, SHAP (KernelExplainer) — with nDCG and top‑k accuracy metrics vs. synthetic ground‑truth masks.


## Installation
Python 3.10+ recommended.

```bash
git clone https://github.com/ElouanV/iterative_learning_for_anomaly_detection.git
cd iterative_learning_for_anomaly_detection
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

Optional: format & lint
```bash
black . && isort . && flake8
```

## Quick Start
Train (Hydra manages outputs under `results/`):
```bash
python src/train_model.py
```
Configuration defaults come from `conf/config.yaml` plus the referenced `defaults` list (dataset, model, training_method). To override inline:
```bash
python src/train_model.py model=DTECategorical dataset=wine training_method=iterative_dataset_sampling random_seed=1
```

Multirun (sweeps over params defined in `hydra.sweeper.params`):
```bash
python src/train_model.py --multirun random_seed=0,1,2 model=DDPM
```

Explain previously trained models (loads weights & metrics, augments them with explanation scores):
```bash
python src/explainability.py
```
Use `conf/config_explainability.yaml` or copy relevant overrides from the training config. Explanations are saved beside the original experiment directory (e.g. `results/<run_id>/<experiment_name>/<dataset>/`).

### Selecting / Scheduling DSIL Ratios
`conf/training_method/iterative_dataset_sampling.yaml` exposes:
```
ratio: 0.5 | cosine | exponential | exponential_v2
sampling_method: deterministic | probabilistic
nu_min / nu_max: min/max kept fraction
max_iter: number of refinement rounds
epoch_budget: if True, per-iteration epochs are reduced proportionally
```

### Models
- `DTECategorical` (`model=DTECategorical` / `model_name: DTEC`): classifies discretized diffusion timesteps.
- `DTEInverseGamma`: inverse-gamma variance modeling.
- `DDPM`: diffusion model with ResNet backbone for reconstruction-based anomaly scoring.

### Synthetic Data Generation
Generate multiple synthetic datasets with embedded anomaly masks:
```bash
python src/dataset/generate_synthetic_data.py
```
Controlled by `conf/config_datagen.yaml`. Output: data arrays, explanation masks, anomaly metadata + an auto-generated dataset config YAML under `conf/dataset/` so the new dataset can be used immediately.

## Configuration (Hydra)
Key high-level fields (`conf/config.yaml`):
- `random_seed` – reproducibility
- `mode` – `benchmark` (full) or `debug` (short epochs / iterations)
- `output_path` – base directory for results
- `run_id` – logical grouping label (e.g., for sweep families)
- `dataset.*` – selected via defaults (`conf/dataset/*.yaml`)
- `model.*` – architecture & training hyperparameters
- `training_method.*` – DSIL / unsupervised / semi-supervised / weighted_loss variants

Each experiment directory stores:
- `experiment_config.yaml` – resolved Hydra config
- `model_metrics.csv` – metrics (F1, AUCROC, timings, explanation quality, ...)
- `train_log.csv` – per-iteration / per-epoch summaries (for iterative methods)
- Model weights (`model.pth`) & auxiliary numpy outputs / plots.

## Explanation Metrics
- `accuracy`: per-sample top‑k (k = ground‑truth sparsity) precision vs. explanation mask.
- `nDCG`: ranking quality normalized by ideal DCG, averaged over explained anomalies.

Ground-truth feature masks come from controlled synthetic anomaly injection (see `generate_synthetic_data.py`). Real datasets may not include `explanation_*` arrays; explanation scripts are intended primarily for synthetic / constructed benchmarks.

## Citing
If you use this code, please cite the paper:
Citation will be available once the paper will be published

## Acknowledgements
Portions of the diffusion / DTE implementations adapt public code from:
- DTE: https://github.com/vicliv/DTE
- Tabular ResNet components: https://github.com/Yura52/rtdl
- Tab-DDPM inspiration: https://github.com/rotot0/tab-ddpm

Please see original repositories for their licenses and cite them where appropriate.

---
Feel free to open issues or discussions.


# Appendix

## ADBench characteristics

Table [f1_score_dataset_info] presents the main characteristics of the datasets included in ADBench.  
Each dataset is described by its name, the number of features, the number of samples, and the percentage of anomalies.  
The datasets vary significantly in size and complexity, with the number of features ranging from 3 to 1555, the number of samples from 80 to 619,326, and the percentage of anomalies from 0.03% to 39.91%.  
This diversity ensures a comprehensive evaluation of anomaly detection methods across different scenarios and challenges.

---

## Additional DSIL results

Table 1 presents the F1 scores for the ADBench datasets, comparing the performance of different variants of the DSIL method (Fixed, Cosine, and Exponential) against DTE-CAT.  
DSIL Fixed shows superior performance in datasets such as "skin," "mammography," "breastw," and "WBC," indicating its effectiveness in certain scenarios.  
DSIL Cosine performs well in datasets like "yeast," "Stamps," and "WDBC," while DSIL Exponential excels in "shuttle," "PageBlocks," and "campaign."  
DTE-CAT demonstrates strong performance in datasets like "smtp," "thyroid," "vertebral," and "annthyroid," among others.  
Overall, DSIL Fixed achieves the best performance in 18 out of the 42 datasets, followed by DTE-CAT with 13, DSIL (ours) Cosine with 11, and DSIL (ours) Exponential with 10.  

---

![AUCROC retention ratio](figures/boxplot_aucroc.png)  
**Figure 1:** AUCROC for different retention ratio of DSIL on 38 ADBench datasets splitted in 4 categories according to retention ratio: [0,1%] (left), [1,2%] (middle left), [2,5%] (middle right), more than 5% (right).

---

![AUCROC over iteration](figures/combined_roc_auc.png)  
**Figure 2:** AUCROC over iteration for different retention rate scheduler on *skin* (left), *WBC* (middle) and *celeba* (right) datasets.

---

## Hyperparameter study

In this section, we explore the impact of hyperparameters of our method, specifically the retention ratio, fixed or with a schedule, and the maximum number of iterations for DSIL.  
When prior knowledge about the number of anomalies in the dataset is available, this information can guide the setting of the retention ratio to closely match the anomaly rate. However, such scenarios are rare because our method is designed for unsupervised learning, where the number of anomalies is typically unknown.  
Therefore, selecting an appropriate retention ratio can be challenging.  

Figure 1 illustrates the mean AUC-ROC of DSIL across 38 datasets for various retention ratio values.  
Due to the wide range of anomaly rates present in these datasets (refer to Table 1 for detailed information), we chose to split them in 4 categories according to the anomaly rate.  
Even though there is no clear best value for the retention ratio, a fixed retention ratio between 0.5 and 0.75 presents the best overall performance when using DSIL.  

Figure 2 illustrates the progression of AUCROC across iterations of the DSIL framework for three real-world datasets. The framework was tested with a fixed retention rate and with retention rates defined by cosine and exponential schedulers. In this experiment, the maximum number of iterations was extended to 15. Iteration 0 represents the results of classic unsupervised training.  

A fixed retention rate of 0.5 enabled rapid convergence within 2 or 3 iterations, with significant improvements observed in the first iteration.  
The exponential scheduler generally underperformed, except on the *celeba* dataset.  
The cosine retention rate scheduler demonstrated a gradual improvement in AUCROC over the iterations, ultimately achieving similar performance to the fixed retention rate by the final iteration and even surpassing it on the *celeba* dataset.  

**Note:** A potential limitation of DSIL is that, if run for too many iterations, the progressive filtering may begin to remove difficult but valid normal samples (“over-purification”), which could impact the generalization. While our experiments did not allow us to directly evaluate this effect, it may contribute to the reduced performance observed on DTE for some datasets.

**Table 1**: Main characteristics of the datasets of ADBench and F1 Scores of our method.

*Bold numbers indicate the best performance per dataset among our method, unsupervised DTE-CAT, and weighted loss iterative learning. Several methods may be considered equally performant if the standard deviation of the top-performing method suggests overlapping performance.*

| Dataset name | #Features | #Samples | %Anomaly | DSIL (ours) Fixed | DSIL (ours) Cosine | DSIL (ours) Exponential | DTE-unsup |
|--------------|-----------|----------|----------|-------------------|--------------------|-------------------------|-----------|
| http         | 3         | 567498   | 0.39     | 0.01±0.01         | **0.02±0.01**      | **0.02±0.01**           | 0.00±0.00 |
| skin         | 3         | 245057   | 20.75    | **0.68±0.00**     | 0.36±0.01          | 0.25±0.01               | 0.21±0.02 |
| smtp         | 3         | 95156    | 0.03     | 0.38±0.04         | 0.38±0.04          | 0.38±0.04               | **0.67±0.03** |
| thyroid      | 6         | 3772     | 2.47     | 0.57±0.02         | 0.66±0.02          | 0.70±0.03               | **0.71±0.03** |
| vertebral    | 6         | 240      | 12.5     | 0.02±0.03         | 0.04±0.02          | 0.04±0.02               | **0.06±0.02** |
| Wilt         | 5         | 4819     | 5.33     | 0.00±0.00         | 0.00±0.00          | 0.00±0.00               | **0.04±0.02** |
| annthyroid   | 6         | 7200     | 7.42     | 0.32±0.00         | 0.36±0.02          | 0.44±0.03               | **0.64±0.02** |
| mammography  | 6         | 11183    | 2.32     | **0.25±0.20**     | 0.20±0.02          | 0.23±0.01               | 0.20±0.00 |
| glass        | 7         | 214      | 4.21     | 0.15±0.05         | 0.13±0.05          | 0.14±0.05               | **0.17±0.02** |
| yeast        | 8         | 1484     | 34.16    | 0.28±0.01         | **0.30±0.00**      | 0.29±0.02               | 0.30±0.01 |
| Pima         | 8         | 768      | 34.90    | **0.51±0.01**     | 0.44±0.02          | 0.44±0.02               | 0.43±0.03 |
| shuttle      | 9         | 49097    | 7.15     | 0.89±0.01         | 0.88±0.02          | **0.92±0.01**           | 0.71±0.04 |
| Stamps       | 9         | 340      | 9.12     | 0.19±0.05         | **0.21±0.05**      | **0.21±0.05**           | **0.21±0.05** |
| breastw      | 9         | 683      | 34.99    | **0.93±0.01**     | 0.77±0.06          | 0.74±0.06               | 0.78±0.03 |
| WBC          | 9         | 223      | 4.48     | **0.72±0.11**     | 0.59±0.24          | 0.35±0.06               | 0.21±0.03 |
| donors       | 10        | 619326   | 5.93     | 0.18±0.12         | **0.18±0.08**      | 0.15±0.09               | 0.13±0.08 |
| cover        | 10        | 286048   | 0.96     | 0.00±0.00         | 0.02±0.01          | 0.04±0.02               | **0.04±0.01** |
| PageBlocks   | 10        | 5393     | 9.46     | 0.38±0.00         | 0.45±0.02          | **0.56±0.02**           | 0.53±0.03 |
| vowels       | 12        | 1456     | 3.43     | 0.47±0.06         | 0.47±0.01          | **0.50±0.04**           | 0.47±0.09 |
| wine         | 13        | 129      | 7.75     | **0.24±0.41**     | 0.03±0.06          | 0.05±0.05               | 0.09±0.11 |
| pendigits    | 16        | 6870     | 2.27     | **0.09±0.06**     | 0.03±0.00          | 0.04±0.01               | 0.04±0.01 |
| Lymphography | 18        | 148      | 4.05     | 0.69±0.16         | **0.73±0.08**      | 0.69±0.16               | 0.48±0.22 |
| Hepatitis    | 19        | 80       | 16.25    | **0.49±0.11**     | 0.31±0.09          | 0.25±0.06               | 0.30±0.08 |
| Cardiotocography | 21    | 2114     | 22.04    | 0.30±0.00         | 0.30±0.01          | **0.31±0.01**           | 0.26±0.02 |
| Waveform     | 21        | 3443     | 2.90     | **0.07±0.01**     | **0.07±0.01**      | 0.07±0.03               | 0.06±0.01 |
| cardio       | 21        | 1831     | 9.61     | **0.56±0.01**     | 0.52±0.02          | 0.32±0.06               | 0.26±0.02 |
| ALOI         | 27        | 49534    | 3.04     | 0.03±0.00         | 0.03±0.00          | 0.03±0.00               | **0.05±0.00** |
| fault        | 27        | 1941     | 34.67    | 0.47±0.03         | **0.47±0.01**      | 0.47±0.03               | 0.45±0.03 |
| fraud        | 29        | 284807   | 0.17     | 0.21±0.06         | 0.24±0.07          | 0.27±0.03               | **0.74±0.02** |
| WDBC         | 30        | 367      | 2.72     | 0.18±0.05         | **0.31±0.18**      | 0.19±0.06               | 0.17±0.06 |
| letter       | 32        | 1600     | 6.25     | 0.28±0.03         | 0.37±0.06          | 0.37±0.06               | **0.36±0.03** |
| WPBC         | 33        | 198      | 23.74    | 0.18±0.04         | 0.18±0.07          | 0.20±0.06               | **0.21±0.03** |
| Ionosphere   | 33        | 351      | 35.90    | **0.86±0.01**     | 0.78±0.03          | 0.77±0.03               | 0.79±0.03 |
| satimage-2   | 36        | 5803     | 1.22     | **0.91±0.02**     | 0.64±0.13          | 0.48±0.10               | 0.15±0.01 |
| satellite    | 36        | 6435     | 31.64    | **0.69±0.01**     | 0.66±0.01          | 0.66±0.01               | 0.61±0.02 |
| landsat      | 36        | 6435     | 20.71    | **0.34±0.03**     | 0.26±0.00          | 0.25±0.01               | 0.20±0.01 |
| celeba       | 39        | 202599   | 2.24     | **0.17±0.01**     | 0.14±0.01          | 0.14±0.01               | 0.09±0.02 |
| SpamBase     | 57        | 4207     | 39.91    | **0.39±0.01**     | 0.39±0.03          | 0.38±0.02               | 0.37±0.02 |
| campaign     | 62        | 41188    | 11.27    | 0.38±0.00         | 0.40±0.03          | **0.43±0.01**           | 0.40±0.01 |
| optdigits    | 64        | 5216     | 2.88     | 0.00±0.00         | 0.00±0.00          | 0.00±0.00               | 0.00±0.00 |
| mnist        | 100       | 7603     | 9.21     | **0.44±0.01**     | 0.38±0.01          | 0.37±0.01               | 0.39±0.04 |
| musk         | 166       | 3062     | 3.17     | **1.00±0.00**     | **1.00±0.00**      | **1.00±0.00**           | 0.51±0.15 |
| backdoor     | 196       | 95329    | 2.44     | 0.51±0.01         | 0.51±0.01          | 0.51±0.01               | **0.51±0.00** |
| speech       | 400       | 3686     | 1.65     | 0.03±0.03         | 0.03±0.02          | **0.03±0.01**           | 0.02±0.01 |
| census       | 500       | 299285   | 6.20     | **0.05±0.00**     | **0.05±0.00**      | **0.05±0.00**           | 0.05±0.01 |
| InternetAds  | 1555      | 1966     | 18.72    | 0.34±0.01         | **0.36±0.00**      | 0.36±0.01               | 0.31±0.06 |


# References:
- [1] M. Kim, J. Yu, J. Kim, T. Oh, and J. K. Choi, “An iterative method for unsupervised robust anomaly detection under data contamination,” IEEETrans. Neur. Net. Learn. Syst., vol. 35, no. 10, pp. 13327–13339, 2024.