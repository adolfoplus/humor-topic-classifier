import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
import time
import os

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(
    page_title="Humor Topic Classifier",
    layout="wide",
    page_icon="😂",
)

# === ESTILO OSCURO PREMIUM ===
dark_style = """
<style>
body {
    background-color: #121212 !important;
}
section.main > div {
    background-color: #121212 !important;
}
.stTitle, .stHeader, h1, h2, h3, h4, h5, h6, label {
    color: #ffffff !important;
}
.stButton button {
    background-color: #00ADB5 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stProgress > div > div {
    background-color: #00ADB5 !important;
}
.sidebar .sidebar-content {
    background-color: #1E1E1E !important;
}
footer {
    visibility: hidden;
}
</style>
"""
st.markdown(dark_style, unsafe_allow_html=True)

# === LOGO & BRANDING HEADER ===
st.markdown(
    """
    <h1 style="color:#00ADB5; font-weight:800; text-align:center;">
        😂 Humor Topic Classifier
    </h1>
    <p style="text-align:center; color:#9E9E9E">
        Zero-Shot BERT • SemEval Humor Task
    </p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "📂 Sube tu archivo CSV/TSV del Task-A", type=["csv", "tsv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.subheader("📊 Datos detectados:")
    st.dataframe(df.head())

    # Limpieza de texto
    def build_text(row):
        if "headline" in df.columns and isinstance(row["headline"], str) and row["headline"].strip() != "-":
            return row["headline"].strip()
        if "word1" in df.columns and "word2" in df.columns:
            return f"{str(row['word1'])} {str(row['word2'])}".strip()
        return ""
    
    df["text_clean"] = df.apply(build_text, axis=1)

    # === Cargando modelo ===
    st.subheader("🧠 Cargando modelo…")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    st.success("✔ Modelo cargado con éxito")

    candidate_labels = [
        "política", "celebridades", "tecnología", "animales", "comida",
        "deportes", "sexo", "crimen", "religión", "salud",
        "trabajo", "dinero", "educación", "familia", "medio ambiente",
        "ciencia", "música", "cine", "internet", "militar"
    ]

    texts = df["text_clean"].tolist()
    total = len(texts)
    batch_size = 32

    topics, scores = [], []

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_box = st.container()
    start_time = time.time()

    output_file = "parcial_clasificacion.csv"

    st.subheader(f"🔄 Procesando {total} textos…")

    try:
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]

            results = classifier(
                batch_texts,
                candidate_labels,
                hypothesis_template="Este texto es sobre {}."
            )
            for r in results:
                topics.append(r["labels"][0])
                scores.append(r["scores"][0])

            df.loc[:len(topics)-1, "topic_bert"] = topics
            df.loc[:len(scores)-1, "topic_score"] = scores
            df.to_csv(output_file, index=False)

            elapsed = time.time() - start_time
            progress = (i + batch_size) / total
            eta = elapsed / progress - elapsed if progress > 0 else 0

            progress_bar.progress(progress)
            status_text.write(
                f"✔ {i+batch_size}/{total} filas • "
                f"{progress*100:.1f}% • ⏱ {elapsed/60:.1f} min • "
                f"ETA: {eta/60:.1f} min"
            )

            with log_box:
                st.markdown(f"🟦 Procesado batch hasta fila **{i+batch_size}**")

            st.download_button(
                "📥 Descargar avance",
                df.to_csv(index=False).encode("utf-8"),
                "clasificacion_parcial.csv",
                "text/csv",
                key=f"partial_{i}"
            )

        status_text.write("🎉 Clasificación completada!")

    except Exception as e:
        st.error(f"❌ Error durante procesamiento: {e}")
        st.warning("Se guardó el progreso parcial")

    # === Resultados ===
    st.subheader("📈 Distribución de temas")

    fig, ax = plt.subplots(figsize=(10,7))
    sns.countplot(
        data=df[df["topic_bert"].notna()],
        y="topic_bert",
        order=df["topic_bert"].value_counts().index,
        ax=ax,
        palette="cool"
    )
    st.pyplot(fig)

    st.subheader("📥 Descargar resultados completos")
    st.download_button(
        "Descargar CSV final",
        df.to_csv(index=False).encode("utf-8"),
        "temas_clasificados_final.csv",
        "text/csv"
    )

# === FOOTER DE MARCA ===
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:#757575; font-size: 14px;'>
        Designed by <b style="color:#00ADB5;">Adolfo Camacho</b>
    </p>
    """,
    unsafe_allow_html=True,
)
