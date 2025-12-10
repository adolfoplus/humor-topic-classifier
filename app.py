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
        model="joeddav/xlm-roberta-large-xnli"
    )

classifier = load_classifier()

st.subheader("📤 Sube tu archivo CSV o TSV")
file = st.file_uploader("Carga un archivo", type=["csv", "tsv"])

if file:
    # Detectar delimitador automáticamente
    delimiter = "\t" if file.type == "text/tab-separated-values" or file.name.endswith(".tsv") else ","
    df = pd.read_csv(file, delimiter=delimiter)

    st.write("📄 Vista previa del archivo:")
    st.dataframe(df.head())

    # 🔍 Detectar la columna de texto
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
        st.error("⚠ No se encontró una columna de texto en el archivo.")
        st.stop()

    st.success(f"🧠 Columna de texto detectada: **{text_col}**")

    texts = df[text_col].astype(str).tolist()

    # Temas definidos
    candidate_labels = ["noticias", "politica", "famosos"]

    resultados = []
    total_texts = len(texts)

    with st.spinner("🔎 Clasificando texto, espera un momento..."):
        for i in range(0, total_texts, 100):
            batch = texts[i:i + 100]
            zsc = classifier(batch, candidate_labels)

            for result in zsc:
                resultados.append(result["labels"][0])

    # Agregar al dataframe
    df["pred_topic"] = resultados

    st.subheader("📊 Gráfica de distribución de temas")
    conteo = Counter(resultados)
    st.write(dict(conteo))

    fig, ax = plt.subplots()
    ax.pie(conteo.values(), labels=conteo.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    st.subheader("📥 Descargar resultados")
    st.dataframe(df.head())
    st.download_button(
        label="⬇ Descargar CSV con predicciones",
        data=df.to_csv(index=False),
        file_name="temas_clasificados.csv",
        mime="text/csv"
    )
