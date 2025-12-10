############################################
# HUMOR TOPIC CLASSIFIER - STREAMLIT APP
############################################

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
from transformers import pipeline

st.set_page_config(page_title="Clasificador de Humor", layout="wide")
st.title("😄 Clasificador de Temas para Task-A (Multilingüe)")

st.write("Zero-shot + humor con estilo mexicano 🇲🇽 (ligero y divertido)")
st.write("Sube tu archivo **TSV** del Task-A 👉 encabezado: `id, word1, word2, headline`")

############################################
# 1️⃣ Subir archivo
############################################
uploaded_file = st.file_uploader("📂 Sube tu archivo TSV aquí", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t")
    st.subheader("👀 Vista previa del archivo")
    st.dataframe(df.head())

############################################
# 2️⃣ Cargar el clasificador (con caché)
############################################
@st.cache_resource
def load_classifier():
    st.write("🤖 Cargando modelo multilingüe Zero-Shot...")
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli",
        device=0 if torch.cuda.is_available() else -1
    )

############################################
# 3️⃣ Clasificar cuando se presione el botón
############################################
if uploaded_file and st.button("🔥 Clasificar temas"):
    classifier = load_classifier()

    # Construir columna de texto limpio
    def clean_text(row):
        if isinstance(row.get("headline"), str) and row["headline"].strip() != "":
            return row["headline"]
        if "word1" in row and "word2" in row:
            return f"{str(row['word1']).strip()} {str(row['word2']).strip()}"
        return ""

    df["text_clean"] = df.apply(clean_text, axis=1)

    # Temas
    candidate_labels = [
        "política", "celebridades", "tecnología", "animales",
        "comida", "deportes", "sexo", "crimen",
        "religión", "salud", "trabajo", "dinero",
        "educación", "familia", "medio ambiente",
        "ciencia", "música", "cine", "internet", "militar"
    ]

    texts = df["text_clean"].tolist()
    topics = []
    scores = []

    progress_bar = st.progress(0)
    total = len(texts)

    st.write(f"⚙️ Procesando {total} ejemplos...")

    for i, text in enumerate(texts):
        result = classifier(
            text,
            candidate_labels,
            hypothesis_template="Este texto es sobre {}."
        )
        topics.append(result["labels"][0])
        scores.append(float(result["scores"][0]))

        progress_bar.progress((i+1)/total)

    df["topic_bert"] = topics
    df["topic_score"] = scores

    st.success("🎉 ¡Clasificación completa!")

    ############################################
    # 4️⃣ Estadísticas y gráficas
    ############################################
    st.subheader("📊 Estadísticas del Corpus")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, y="topic_bert", order=df["topic_bert"].value_counts().index)
    plt.title("Distribución de Temas Detectados")
    plt.xlabel("Cantidad")
    plt.ylabel("Tema")
    st.pyplot(fig)

    ############################################
    # 5️⃣ Descargar resultado
    ############################################
    output_name = "clasificacion_humor_completa.csv"
    csv = df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="📥 Descargar resultados en CSV",
        data=csv,
        file_name=output_name,
        mime="text/csv"
    )
