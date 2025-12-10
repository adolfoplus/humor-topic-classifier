import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import Counter

# ======================
# CARGA DEL MODELO
# ======================
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

classifier = load_classifier()

# Temas a clasificar
TOPICS = ["noticias", "política", "famosos", "deportes", "humor", "tecnología"]

# ======================
# STREAMLIT UI
# ======================
st.title("📊 Clasificador de Temas")
st.write("Procesa textos en lotes de 100 y muestra una gráfica de pastel.")

file = st.file_uploader("📥 Sube un archivo .tsv o .csv", type=["csv", "tsv"])

if file:
    sep = "\t" if file.name.endswith(".tsv") else ","
    df = pd.read_csv(file, sep=sep)

    if "text" not in df.columns:
        st.error("El archivo debe tener una columna llamada 'text'")
        st.stop()

    texts = df["text"].tolist()
    total = len(texts)

    batch_size = 100
    results = []

    progress_bar = st.progress(0)

    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        output = classifier(batch, TOPICS, multi_label=False)

        for o in output:
            topic = o["labels"][0] if o["labels"] else "otros"
            results.append(topic)

        progress_bar.progress(min((i+batch_size)/total, 1.0))

    df["topic"] = results

    st.success("¡Clasificación completada!")
    st.dataframe(df.head(20))

    # 📊 Conteo por tema
    counts = Counter(results)
    st.subheader("Distribución de temas")

    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # Descargar resultado
    st.download_button(
        "📎 Descargar resultados",
        df.to_csv(index=False),
        "resultados.csv",
        "text/csv"
    )
