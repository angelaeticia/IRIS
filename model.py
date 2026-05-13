from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Chargement du dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modèle
model = RandomForestClassifier()

# Entraînement
model.fit(X_train, y_train)

# Sauvegarde
pickle.dump(model, open("model.pkl", "wb"))

print("Modèle entraîné avec succès")