import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clasificador de Temas", layout="wide")

st.title("📌 Clasificación de Temas")

@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli-pruned",
        use_fast=False  # 🔥 Solución clave para evitar el error
    )

classifier = load_classifier()

st.subheader("📤 Sube tu archivo CSV o TSV")
file = st.file_uploader("Carga un archivo", type=["csv", "tsv"])

if file:
    # Detectar delimitador
    delimiter = "\t" if file.name.endswith(".tsv") else ","
    df = pd.read_csv(file, delimiter=delimiter)

    st.write("📄 Vista previa del archivo:")
    st.dataframe(df.head())

    # 🔍 Detectar columna de texto
    text_col = None
    columnas_ignorar = {"id", "word1", "word2", "topic", "score", "joke"}

    for col in df.columns:
        if col.lower() == "text":
            text_col = col
            break
        if col.lower() not in columnas_ignorar:
            text_col = col
            break

    if text_col is None:
        st.error("⚠ No se encontró una columna de texto.")
        st.stop()

    st.success(f"🧠 Columna de texto detectada: **{text_col}**")

    texts = df[text_col].astype(str).tolist()

    labels = ["noticias", "politica", "famosos"]

    resultados = []
    total = len(texts)

    with st.spinner("🔎 Clasificando..."):
        for i in range(0, total, 100):
            batch = texts[i:i + 100]
            zsc = classifier(batch, labels)
            for r in zsc:
                resultados.append(r["labels"][0])

    df["pred_topic"] = resultados

    st.subheader("📊 Distribución de temas")
    conteo = Counter(resultados)
    fig, ax = plt.subplots()
    ax.pie(conteo.values(), labels=conteo.keys(), autopct="%1.1f%%")
    st.pyplot(fig)

    st.subheader("📥 Resultado")
    st.dataframe(df.head())
    st.download_button(
        "⬇ Descargar CSV",
        df.to_csv(index=False),
        "temas_clasificados.csv",
        "text/csv"
    )
