import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# =====================================================
#          CARGA DEL MODELO DE CLASIFICACIÓN
# =====================================================
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="Recognai/bert-base-multilingual-uncased"
    )

classifier = load_classifier()

# Temas a clasificar
TOPICS = ["noticias", "política", "famosos"]


# =====================================================
#                INTERFAZ DE LA APP
# =====================================================
st.title("🧠 Clasificador de Temas (Batch 100)")
st.markdown("Clasifica textos en **noticias**, **política** y **famosos** por lotes de 100.")

uploaded_file = st.file_uploader("📄 Sube tu archivo CSV o TSV", type=["csv", "tsv"])

if uploaded_file is not None:

    # Detecta automáticamente CSV o TSV
    if uploaded_file.name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t", on_bad_lines="skip")
    else:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip")

    if "text" not in df.columns:
        st.error("❌ El archivo debe contener una columna llamada **text**")
        st.stop()

    st.success("📂 Archivo cargado correctamente")
    st.write(df.head())

    if st.button("🚀 Procesar clasificación"):
        st.info("⏳ Procesando textos…")  

        results = []
        batch_size = 100
        total = len(df)
        progress = st.progress(0)

        # Procesar por bloques de 100
        for i in range(0, total, batch_size):
            batch = df["text"][i:i+batch_size].tolist()

            for text in batch:
                zsc = classifier(text, TOPICS)
                topic = zsc["labels"][0]
                score = float(zsc["scores"][0])
                results.append((topic, score))

            progress.progress(min(1, (i+batch_size)/total))

        df["topic"] = [r[0] for r in results]
        df["score"] = [r[1] for r in results]

        st.success("✨ Clasificación completada")

        # =============================
        #    GRÁFICA DE PASTEL
        # =============================
        st.subheader("📊 Distribución de temas")
        counts = df["topic"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # Descargar resultados
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar resultados",
            data=csv_out,
            file_name="resultados_clasificados.csv",
            mime="text/csv"
        )
