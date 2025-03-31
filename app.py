import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(page_title="Classificador Gats vs Gossos", layout="centered")
st.title("🐶 Classificador de Gossos i Gats 🐱")
st.markdown("Puja una imatge i la IA et dirà si veu un gos o un gat! 🧠")

uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption='📷 Imatge pujada', use_column_width=True)

    model = load_model('model_gats_gossos.h5')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = float(prediction[0])

    if prob > 0.5:
        st.success(f"És un **gos** 🐶 amb {prob*100:.2f}% de confiança!")
    else:
        st.success(f"És un **gat** 🐱 amb {(1 - prob)*100:.2f}% de confiança!")
