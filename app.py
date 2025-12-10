import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# =====================================================
#   CARGA DEL MODELO DE CLASIFICACIÓN (ZERO-SHOT)
# =====================================================
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual"
    )

classifier = load_classifier()

TOPICS = ["noticias", "política", "famosos"]

# =====================================================
#                INTERFAZ
# =====================================================
st.title("🧠 Clasificador de Temas")
st.write("Clasifica textos en noticias, política o famosos por bloques de 100.")

uploaded_file = st.file_uploader("📄 Subir archivo CSV o TSV", type=["csv", "tsv"])

if uploaded_file is not None:

    if uploaded_file.name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t", on_bad_lines="skip")
    else:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip")

    if "text" not in df.columns:
        st.error("❌ El archivo debe tener una columna llamada 'text'")
        st.stop()

    st.write(df.head())

    if st.button("🚀 Procesar archivo"):
        batch_size = 100
        total = len(df)
        st.info(f"Procesando {total} textos...")
        progress = st.progress(0)

        topics_out = []
        scores_out = []

        for i in range(0, total, batch_size):
            batch = df["text"][i:i+batch_size].tolist()

            for text in batch:
                result = classifier(text, TOPICS)
                topics_out.append(result["labels"][0])
                scores_out.append(float(result["scores"][0]))

            progress.progress(min(1.0, (i + batch_size) / total))

        df["topic"] = topics_out
        df["score"] = scores_out

        st.success("🎯 Clasificación lista")

        # 📊 GRÁFICA DE PASTEL
        st.subheader("Distribución de temas")
        counts = df["topic"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # Descargar CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar resultados",
            csv,
            "clasificacion_tematicas.csv",
            "text/csv"
        )
