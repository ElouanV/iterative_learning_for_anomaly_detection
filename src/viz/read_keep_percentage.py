import matplotlib.pyplot as plt
import numpy as np

# Ouvrir le fichier et lire son contenu
with open("resultats_keep_percentage.csv", "r") as file:
    lines = file.readlines()

# Initialiser les listes pour chaque colonne
dataset_names = []
f1_semi = []
f1_unsup = []
f1_3 = []
f1_4 = []
f1_5 = []
f1_6 = []
f1_7 = []
f1_8 = []
f1_9 = []

# Parcourir chaque ligne (en ignorant les lignes de header et de séparation)
for line in lines[2:]:
    # Enlever les espaces en début et fin de ligne et séparer par '|'
    parts = line.strip().split("|")
    if len(parts) == 10:
        dataset_names.append(parts[0].strip())
        f1_unsup.append(float(parts[1].strip()))
        f1_semi.append(float(parts[2].strip()))
        f1_3.append(float(parts[3].strip()))
        f1_4.append(float(parts[4].strip()))
        f1_5.append(float(parts[5].strip()))
        f1_6.append(float(parts[6].strip()))
        f1_7.append(float(parts[7].strip()))
        f1_8.append(float(parts[8].strip()))
        f1_9.append(float(parts[9].strip()))

num_dataset = len(dataset_names)

f1_unsup = np.array(f1_unsup)
f1_semi = np.array(f1_semi)
f1_3 = np.array(f1_3)
f1_4 = np.array(f1_4)
f1_5 = np.array(f1_5)
f1_6 = np.array(f1_6)
f1_7 = np.array(f1_7)
f1_8 = np.array(f1_8)
f1_9 = np.array(f1_9)

# Créer le dictionnaire avec les listes
data_dict = {
    "dataset_names": dataset_names,
    "f1_unsup": f1_unsup,
    "f1_semi": f1_semi,
    "f1_3": f1_3,
    "f1_4": f1_4,
    "f1_5": f1_5,
    "f1_6": f1_6,
    "f1_7": f1_7,
    "f1_8": f1_8,
    "f1_9": f1_9,
}

mean_std_dict = {}

for key, value in data_dict.items():
    if (
        key != "dataset_names"
    ):  # Ignorer les noms de dataset car ce ne sont pas des valeurs numériques
        mean_value = np.mean(value)
        std_value = np.std(value)
        mean_std_dict[key] = (mean_value, std_value)
        print(
            f"{key}: Moyenne = {mean_value:.2f}, Ecart-type = {std_value:.2f}"
        )


# Trier par ordre décroissant de moyenne
sorted_methods = sorted(
    mean_std_dict.items(), key=lambda item: item[1][0], reverse=False
)

# Préparer les données pour le graphique
methods = [item[0] for item in sorted_methods]
means = [item[1][0] for item in sorted_methods]
stds = [item[1][1] for item in sorted_methods]

plt.figure(figsize=(10, 8))

for i, (mean, std) in enumerate(zip(means, stds)):
    plt.barh(
        methods[i], width=2 * std, height=0.4, left=mean - std, color="grey"
    )
    plt.vlines(mean, ymin=i - 0.2, ymax=i + 0.2, color="black", linewidth=2)

plt.xlabel("F1 Score")
plt.ylabel("Methods")
plt.title("F1 Scores with Error Bars Representing Standard Deviation")
plt.grid(True)
plt.xlim(0, max(means) + max(stds))
plt.savefig("./percentage_keep_results.png")
plt.show()
