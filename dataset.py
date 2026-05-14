from sklearn.datasets import load_iris
import pandas as pd

# Chargement du dataset
iris = load_iris()

# Création d'un DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Ajout de la colonne cible (espèce)
df["species"] = iris.target
df["species"] = df["species"].map(dict(enumerate(iris.target_names)))

# Sauvegarde
df.to_csv("iris_dataset.csv", index=False)