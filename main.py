from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Création API
app = FastAPI(title="API de Classification Iris")

# Modèle
model = joblib.load("model.pkl")

# Structure des données reçues
class IrisFlower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint Accueil
@app.get("/")
def accueil():
    return {
        "message": "Bienvenue sur l'API de classification Iris"
    }

# Endpoint Prédiction
@app.post("/predict")
def predict(flower: IrisFlower):

    data = pd.DataFrame([[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ])

    prediction = model.predict(data)[0]

    species = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    predicted_class = species[prediction]

    return {
        "predicted_flower": predicted_class
    }

# Vérification santé API
@app.get("/health")
def health_check():
    return {
        "status": "API fonctionnelle",
        "model_loaded": True if model else False
    }