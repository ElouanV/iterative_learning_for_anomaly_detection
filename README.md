# Install
In your environment:
```sh
pip install -r requirements.txt
```

# Train a model
```sh
python src/train_model.py
```
will run a model training using configuration in `conf/config.yaml` and configuration subdirectories
You can run several configuration in one command using argument `--multirun` and modifying section `hydra.sweeper.params` in `config.yaml`.

# Explain a model
You can run 
```sh
python src/explainability.py
```
that will use `config_explainability.yaml`.
If you want to explain the model you just trained, copy the content of `config.yaml` to `config_explainability.yaml`

# Generate synthetic dataset
```sh
python src/dataset/generate_synthetic_data.py
```
It will use the `conf/config_datagen.yaml`.

You can also run
```sh
python src/dataset/adbench_synthetic_anomalies.py
```
Both scripts will generate datasets and the configuration file to train a model on it, stored in `conf/dataset`

# Post-process
In `notebooks/`, you will find different notebooks to extract data from the log of the training. Allowing to plot what is presented in the paper. To ensure anonymity, outputs of cells are not included in the repository. You can run the notebooks and save the outputs yourself.