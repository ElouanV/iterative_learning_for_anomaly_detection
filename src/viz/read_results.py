import numpy as np

# Ouvrir le fichier et lire son contenu
with open("resultats_simulation.csv", "r") as file:
    lines = file.readlines()
# Initialiser les listes pour chaque colonne
dataset_names = []
f1_semi_supervised = []
f1_unsupervised = []
f1_itered = []

# Parcourir chaque ligne (en ignorant les lignes de header et de séparation)
for line in lines[3:]:
    # Enlever les espaces en début et fin de ligne et séparer par '|'
    parts = line.strip().split("|")
    if len(parts) == 4:
        dataset_names.append(parts[0].strip())
        f1_semi_supervised.append(float(parts[1].strip()))
        f1_unsupervised.append(float(parts[2].strip()))
        f1_itered.append(float(parts[3].strip()))

num_dataset = len(dataset_names)

f1_semi_supervised = np.array(f1_semi_supervised)
f1_unsupervised = np.array(f1_unsupervised)
f1_itered = np.array(f1_itered)


# Créer la liste delta = f1_itered_max - f1_unsupervised_max
delta = f1_itered - f1_unsupervised

# Calculer la moyenne et l'écart type pour chaque métrique
mean_f1_semi_supervised = np.mean(f1_semi_supervised)
std_f1_semi_supervised = np.std(f1_semi_supervised)

mean_f1_unsupervised = np.mean(f1_unsupervised)
std_f1_unsupervised = np.std(f1_unsupervised)

mean_f1_itered = np.mean(f1_itered)
std_f1_itered = np.std(f1_itered)

mean_delta = np.mean(delta)
std_delta = np.std(delta)

# Afficher les résultats
print(
    f"Moyenne et écart type de f1_semi_supervised : {mean_f1_semi_supervised:.3f}, {std_f1_semi_supervised:.3f}"
)
print(
    f"Moyenne et écart type de f1_unsupervised : {mean_f1_unsupervised:.3f}, {std_f1_unsupervised:.3f}"
)
print(
    f"Moyenne et écart type de f1_itered : {mean_f1_itered:.3f}, {std_f1_itered:.3f}"
)
print(f"Moyenne et écart type de delta : {mean_delta:.3f}, {std_delta:.3f}")
