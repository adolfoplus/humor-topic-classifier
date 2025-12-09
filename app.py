import streamlit as st
import pandas as pd
import time
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# =========================
# App Config
# =========================
st.set_page_config(page_title="Humor Topic Classifier", layout="wide")
st.title("🎯 Humor Topic Classifier")

st.write("Clasificación de titulares en 3 temas: **noticias**, **famosos** y **política**")
st.write("Procesamiento en lotes de 100 ejemplos")

# =========================
# Load Embedding model
# =========================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embedder = load_embedder()

# =========================
# File Upload
# =========================
uploaded = st.file_uploader("📥 Cargar archivo CSV/TSV del Task-A", type=["csv", "tsv"])

if uploaded:
    df = pd.read_csv(uploaded, sep="\t" if uploaded.name.endswith(".tsv") else ",")
    st.subheader("Vista previa del archivo")
    st.dataframe(df.head(5))

    if st.button("🚀 Iniciar Clasificación"):
        st.subheader("Procesamiento en curso...")
        topics = ["política", "famosos", "noticias"]
        topic_vectors = embedder.encode(topics)

        batch_size = 100
        results = []

        progress = st.progress(0)
        status = st.empty()

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            texts = batch["headline"].tolist()

            embeddings = embedder.encode(texts)
            sims = cosine_similarity(embeddings, topic_vectors)

            for j, _ in enumerate(texts):
                topic_index = sims[j].argmax()
                results.append({
                    "id": batch["id"].iloc[j],
                    "text": batch["headline"].iloc[j],
                    "topic": topics[topic_index],
                    "score": float(sims[j][topic_index])
                })

            progress.progress(min((i+batch_size)/len(df), 1.0))
            status.text(f"Procesados: {min(i+batch_size, len(df))}/{len(df)}")
            time.sleep(0.5)

        st.success("📌 Clasificación finalizada correctamente")
        result_df = pd.DataFrame(results)

        st.subheader("📊 Distribución de Temas")
        counts = result_df["topic"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Distribución de temas en el dataset")
        st.pyplot(fig)

        st.subheader("📄 Resultados")
        st.dataframe(result_df.head(10))

        st.download_button(
            "📁 Descargar resultados (CSV)",
            data=result_df.to_csv(index=False),
            file_name="resultados_clasificacion.csv"
        )

# =========================
# Footer
# =========================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Aplicación desarrollada por Adolfo Camacho — turboplay333@gmail.com")
