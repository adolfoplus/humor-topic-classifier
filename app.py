import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# ============================
#    Cargar modelo una vez
# ============================
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual"  # ✔ Modelo estable
    )


classifier = load_classifier()
TOPICS = ["noticias", "política", "famosos"]


# ============================
#        Interfaz
# ============================
st.title("🧠 Clasificador Temático de Titulares")
st.write("Carga tu archivo y clasifica los titulares en tres temas: noticias, política o famosos.")

uploaded = st.file_uploader("📄 Subir archivo CSV/TSV", type=["csv", "tsv"])

if uploaded is not None:
    # Leer CSV o TSV
    if uploaded.name.endswith(".tsv"):
        df = pd.read_csv(uploaded, sep="\t", on_bad_lines="skip")
    else:
        df = pd.read_csv(uploaded, on_bad_lines="skip")

    if "text" not in df.columns:
        st.error("❌ El archivo debe contener una columna llamada **text**.")
        st.stop()

    st.write("📌 Vista previa de tus datos:")
    st.dataframe(df.head())

    if st.button("🚀 Clasificar titulares"):
        batch_size = 100
        total = len(df)
        st.info(f"Procesando {total} textos…")
        progress = st.progress(0)

        topics = []
        scores = []

        for i in range(0, total, batch_size):
            batch = df["text"][i:i+batch_size].tolist()

            for text in batch:
                try:
                    res = classifier(text, TOPICS)
                    topics.append(res["labels"][0])
                    scores.append(float(res["scores"][0]))
                except Exception:
                    topics.append("desconocido")
                    scores.append(0.0)

            progress.progress(min(1.0, (i + batch_size) / total))

        df["topic"] = topics
        df["score"] = scores

        st.success("🎯 Clasificación completada")

        # ============================
        #       Gráfica de pastel
        # ============================
        st.subheader("📊 Distribución de temas")
        topic_counts = df["topic"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # ============================
        #    Botón de descarga
        # ============================
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar CSV con resultados",
            csv,
            file_name="clasificacion_tematicas.csv",
            mime="text/csv"
        )
