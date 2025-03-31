# 📁 app.py – STREAMLIT
# ✅ Versió final per carregar model optimitzat (JSON + WEIGHTS)

import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np

st.set_page_config(page_title="Classificador Gats vs Gossos", layout="centered")
st.title("🐶 Classificador de Gossos i Gats 🐱")
st.markdown("Puja una imatge i la IA et dirà si veu un gos o un gat! 🧠")

uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption='📷 Imatge pujada', use_container_width=True)

    # Carregar arquitectura
    with open("model_gats_gossos.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)

    # Carregar pesos
    model.load_weights("model_gats_gossos.weights.h5")

    # Preprocessar imatge
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicció
    prediction = model.predict(img_array)
    prob = float(prediction[0])

    if prob > 0.5:
        st.success(f"És un **gos** 🐶 amb {prob*100:.2f}% de confiança!")
    else:
        st.success(f"És un **gat** 🐱 amb {(1 - prob)*100:.2f}% de confiança!")
