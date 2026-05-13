import streamlit as st
import pickle
import numpy as np

# Charger modèle
model = pickle.load(open("model.pkl", "rb"))

st.title("Classification Iris")

st.write("Entrez les caractéristiques de la fleur")

sepal_length = st.slider("Longueur des sépales", 4.0, 8.0, 5.0)
sepal_width = st.slider("Largeur des sépales", 2.0, 5.0, 3.0)
petal_length = st.slider("Longueur des pétales", 1.0, 7.0, 4.0)
petal_width = st.slider("Largeur des pétales", 0.1, 3.0, 1.0)

features = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])

if st.button("Prédire"):

    prediction = model.predict(features)

    species = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(f"Typologie de fleur prédite : {species[prediction[0]]}")