# Guide d'utilisation itered_DTE

### Code de base

Tout d'abord le modèle sous-jacent sur lequel on itère est codé dans le  fichier [dte.py](./dte.py).

Ce code est issus du repo suivant https://github.com/vicliv/DTE qui implémente le modèle décrit dans le papier [On Diffusion Modeling for Anomaly Detection](https://openreview.net/forum?id=lR3rk7ysXz&noteId=lR3rk7ysXz) (V. Livernoche, V. Jain, Y. Hezaveh, S. Ravanbakhsh)


### Données

#### Données Réelles
Les données [data](./data) proviennent d'Adbench, un benchmark courramment utilisé pour la détection d'anomalie dans des données tabulaires. [ADBench: Anomaly Detection Benchmark](https://arxiv.org/abs/2206.09426) (S. Han, X. Hu, H. Huang, M. Jiang, Y. Zhao)


#### Données Synthétiques
Dans l'objectif de faire de l'explicabilité, un code supplémentaire pour générer des données synthétique sur lesquelles on maîtrise les anomalies a été produit : [generate_data.py](./generate_data.py).

**[generate_data.py](./generate_data.py)**
Ce code créer des datasets contenant 8 features et une colonne anomalie. Les features sont générés à partir de lois normales auxquelles on ajoute 3 types d'anomalies :
- altérer les hyperparamètres de certaines variables (appliquable sur la feature 1)
- altérer les directions de causalité (enlever ou ajouter des corrélation) (appliquable sur les features 2,3,5)
- Ajouter des variables latentes (connecté aux variable observé mais invisible dans le jeu de données) (appliquable sur les features 3,4)



**Paramètres**
* "-p", "--dataSetPath" : Chemin où enregistrer le dataset synthétique généré
* "-n",'--num_sample' : Nombre de ligne dans le dataset
* "-a1",'--prop_anomalies1' : Proportion d'anomalie du premier type dans le dataset
* "-a2",'--prop_anomalies2' : Proportion d'anomalie du second type dans le dataset
* "-a3",'--prop_anomalies3' : Proportion d'anomalie du troisième type dans le dataset
* "-c", "--add_correlation" : Ajoute une corrélation de base entre les features 3 et 5

<ins>*Exemple*</ins>
```
$ python .\generate_data.py -p "./data_synth/synth_data1.csv" -n 1200 -a1 0.05 -a3 0.03
```

Cela génère le fichier synth_data1.csv dans le dossier [./data_synth](./data_synth/), ce dataset contient 1200 observation comprennant 5% d'anomalie de type 1 et 3% d'anomalie de type 3, et pas de corrélation.



### Modèle itéré
Le coeur de mon travail se situe dans le fichier [dte_itered.py](./dte_itered.py)
Ce code contient les deux pans de mon travail, à savoir :
- L'implémentation du système d'itération en choisissant judicieusement les données d'entrainement à chaque itérations
- L'implémentation du bruitage sélectif pour expliquer quelles variables sont impactés par le modèle DTE. Le module shap dont la documentation est consultable [ici](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) est utilisé comme baseline pour se comparé en terme d'explicabilité.

*Remarque* : le temps de calcule est considérablement augmenté lorsqu'on performe le bruit sélectif et le calcule des shapley values. Ces données ne servant qu'à faire de l'explicabilité, si on veut simplement calculer des performances de score F1, il ne vaut mieux pas les utilisés.

**Paramètres**
- "-d", "--dataSet" : Chemin du dataset à traité
- "-n", "--selective_noise" : à ajouter si on veut calculer le bruité sélectif sur toutes les variables de ce dataset
- "-s", "--shapley_values" : à ajouter si on veut calculer les scores par shapley values sur toutes les variables de ce dataset
- "-p", "--perc_keep" : à ajouter si on veut comparer les performances pour différentes proportion du dataset utilisés pour l'entrainement. /!\ temps de calcule x7 car les itérations sont refaites 7 fois pour les 7 pourcentages allant de 30% à 90%.

<ins>*Exemples*</ins>

(i) 
```
$ python dte_itered.py -d"./data_adbench/4_breastw.npz"
```
this will perform iterations with dte on the dataset 4_breastw of ADBench. That will ultimately produce : 
1) the figure [summary_p_0.5.png](./res/4_breastw/summary_p_0.5.png) in dir ./res/4_breastw/. This figure represent the evolution of F1 and training loss over iterations with 50% of dataset used in training.

2) This will store the following results : 4_breastw | F1 semi-supervised |   F1 unsupervised  | F1 itered, with a new line in [./resultat_simulation.csv](./resultat_simulation.csv)

(ii) 

```
$ python dte_itered.py -d"./data_adbench/32_shuttle.npz" -p
```
This will perfom iteration with dte on 32_shuttle with percentage_keep=0.3,0.4,...0.9. That will ultimately produce :
1) The seven figures summary_p_0.3.png, summary_p_0.4.png, ... summary_p_0.9.png in dir ./res/32_shuttle. These figures will show the evolution of F1 and Training Loss over iteration for the different percentage_keep.

2) This will stores the following results : 32_shuttle  | F1 (semi) | F1 (unsup) | F1 (p = 0.3) | F1 (p = 0.4) | F1 (p = 0.5) | F1 (p = 0.6) | F1 (p = 0.7) | F1 (p = 0.8) | F1 (p = 0.9) with a new line in [./resultats_keep_percentage.csv](./resultats_keep_percentage.csv)

(iii)
```
python dte_itered.py -d"./data_synth/synth_data1.csv" -n -s
```
This will perfom iteration with dte on the synthetic dataset synth_data1. Then the trained model will be analysed using selective noise (-n) and shapley values (-s). That will ultimately produce :

1) the figure summary_p_0.5.png in dir ./res/synth_data1/. This figure represent the evolution of F1 and training loss over iterations with 50% of dataset used in training.
2) This will store the following results : synth_data1 | F1 semi-supervised |   F1 unsupervised  | F1 itered, with a new line in [./resultat_simulation.csv](./resultat_simulation.csv)
3) selective_noise.png and selective_noise_scores.txt : the score of each features obtained by noising them before doing predictions (figures barplot + values stores in .txt). And couple_noise.png that will be the matrix obtained by noising couple of features
4) /shap_values.png and /shap_scores.txt : the score of each features obtained using shapley values (figures barplot + values stores in .txt).


### Automatisation des processus
Pour obtenir des résultats fiables, il faut faire tourner le modèle d'itération sur plusieurs datasets. C'est pour cela que le script [run_simulation.py](./run_simulation.py) a été conçu.

**Paramètres**
- "-f", "--data_folder" : chemin vers le dossier contenant les datasets qu'on veut utilisés pour tester notre modèle
- "-n", "--selective_noise" : booléen à ajouter si on veut calculer les résultats des bruits sélectifs sur chaque dataset /!\ temps de caclul bien plus important : Intérêt restreint à de l'explicabilité
- "-s", "--shapley_values" : booléen à ajouter si on veut calculer des bruits sélectifs sur chaque dataset /!\ temps de caclul bien plus important : Intérêt restreint à de l'explicabilité
- "-p", "--perc_keep" : booléen à ajouter si on veut effectuer le calcule pour des sets d'entrainent de différentes tailles : 30%, 40%, ... 90% du jeu de données complet. /!\ x7 sur le temps de calcul

<ins>*Exemples*</ins>
(i)
```
$ python run_simulation -d"./data_adbench"
```

applique dte_itered sur tous les datasets dans le dossier [data_adbench](./data_adbench/). Cela fournira notamment les scores F1 pour tous les datasets en faisant semi-supervisé, non-supervisé et itéré dans [resultats_simulation.csv](./resultats_simulation.csv), et également si dataset_name est le nom d'un dataset dans le dossier alors les figures montrant l'évolution au cours des itérations seront présentes dans le dossier res/dataset_name

(ii)
```
$ python run_simulation -d"./data_adbench" -p
```
applique dte_itered sur tous les datasets dans le dossier [data_adbench](./data_adbench/) et pour percentage_keep variant de 0.3 à 0.9. Cela fournira notamment les scores F1 pour tous les datasets et pour les différents percentage_keep dans [resultats_keep_percentage.csv](./resultats_keep_percentage.csv), et également si dataset_name est le nom d'un dataset dans le dossier alors les figures montrant l'évolution au cours des itérations pour p variant de 0.3 à 0.9 seront présentes dans le dossier res/dataset_name


(iii)
```
$ python run_simulation -d"./data_synth" -n -s
```
applique dte_itered sur tous les datasets dans le dossier [data_synth](./data_synth/), on calculera également les scores des différentes features en utilisant le bruitage sélectif et les shapley_values. En plus des résultats comme précédemment stockées dans [resultats_simulation.csv](./resultats_simulation.csv) et les figures d'évolution au cours des itérations dans res/dataset_name, on aura également dans res/dataset_name les figures affichant les scores par bruitage sélectif et shapley values sous forme de barplot, ainsi que la heatmap représentant le bruitage sélectif par couple.

### Lecture des résultats

Les scripts [read_results.py](./read_results.py) et [read_keep_percentage.py](./read_keep_percentage.py) servent à lire les résultats des fichiers [resultats_simulation.csv](./resultats_simulation.csv) et [resultats_keep_percentage.csv](./resultats_keep_percentage.csv). Ils s'exécutent sans argument simplement avec :
```
$ python read_results.py
$ python read_keep_percentage
```
Le premier script [read_results.py](./read_results.py) caclule simplement les moyennes et écart-types des différents F1 (F1 semi-supervised |   F1 unsupervised  | F1 itered) pour les différents datasets. La valeur ajouté de l'itération ; $\Delta = $ F1_itered  - F1_unsupervised ; est également mesuré.

Le second script [read_keep_percentage.py](./read_keep_percentage.py) calcules les moyennes et écart-types des différents F1 avec des percentage_keep différents. Un graphique est ensuite tracé pour voir quel percentage_keep est optimal sur les datasets testés.
___


Pour poursuivre le travail dans l'objectif d'une publication :

### Todo list


- **Use NDCG**: Apply Normalized Discounted Cumulative Gain (NDCG) to compare algorithms in terms of their ability to estimate which variables are affected by noise in anomaly points.

- **Enhanced ADBench**: Improved ADBench to identify which variables are affected for each anomaly point in a given dataset.

- **Duality in Anomaly Detection**: There is a dual perspective in anomaly detection: (1) from the point perspective and (2) from the feature perspective. The second aspect, focusing on feature perspective, has not yet been addressed by ADBench.
