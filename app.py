import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

st.set_page_config(page_title="Clasificación Humorística", layout="wide")
st.title("😂 Clasificación de Temas Humorísticos (BERT Zero-Shot)")

uploaded_file = st.file_uploader("📂 Sube tu archivo CSV/TSV del Task-A", type=["csv", "tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.write("📊 Datos detectados:", df.head())

    def clean_text(row):
        if "headline" in df.columns and isinstance(row.get("headline"), str) and row["headline"].strip() != "":
            return row["headline"].strip()
        if "word1" in df.columns and "word2" in df.columns:
            return f"{str(row['word1']).strip()} {str(row['word2']).strip()}"
        return ""

    df["text_clean"] = df.apply(clean_text, axis=1)

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

    candidate_labels = [
        "política", "celebridades", "tecnología", "animales",
        "comida", "deportes", "sexo", "crimen",
        "religión", "salud", "trabajo", "dinero",
        "educación", "familia", "medio ambiente",
        "ciencia", "música", "cine", "internet", "militar"
    ]

    texts = df["text_clean"].tolist()
    topics, scores = [], []
    progress = st.progress(0)

    for i in range(len(texts)):
        result = classifier(
            texts[i],
            candidate_labels,
            hypothesis_template="Este texto es sobre {}."
        )
        topics.append(result["labels"][0])
        scores.append(result["scores"][0])
        progress.progress((i+1)/len(texts))

    df["topic_bert"] = topics
    df["topic_score"] = scores

    st.subheader("📈 Distribución de temas")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, y="topic_bert", ax=ax)
    st.pyplot(fig)

    st.download_button(
        "📥 Descargar CSV con resultados",
        df.to_csv(index=False).encode("utf-8"),
        "temas_clasificados.csv",
        "text/csv"
    )

    st.success("🎉 Procesamiento completado correctamente!")
