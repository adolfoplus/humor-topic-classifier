import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Humor Topic Classifier",
    page_icon="🤣",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= ESTILOS PREMIUM OSCUROS =======
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #FAFAFA;
}
.block-container {
    padding-top: 2rem;
}
.stProgress > div > div {
    background-color: #29B6F6 !important;
}
.footer {
    text-align: center;
    padding: 15px;
    color: #888;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ======= ENCABEZADO =======
st.title("🤣 Humor Topic Classifier")
st.caption("Zero-Shot BERT + Generación de Chistes")

uploaded_file = st.file_uploader(
    "📂 Sube tu archivo CSV/TSV del Task-A",
    type=["csv", "tsv"]
)

resume = st.checkbox("🔄 Reanudar desde progress_partial.csv (si existe)", value=True)


# ======= CARGA DE MODELOS =======
def load_zero_shot():
    return pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1)

def load_joke_model():
    return pipeline("text-generation",
                    model="gpt2",
                    device=0 if torch.cuda.is_available() else -1)


if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep="\t")

    st.subheader("👀 Vista previa")
    st.dataframe(df.head())

    df["topic"] = None
    df["score"] = None
    df["joke"] = None

    # Reanudar si existe
    if resume and os.path.exists("progress_partial.csv"):
        prev = pd.read_csv("progress_partial.csv")
        df.update(prev)

    st.subheader("⚙️ Cargando modelos de IA…")
    status = st.empty()
    status.info("🔍 Cargando modelo Zero-Shot (rápido)…")
    classifier = load_zero_shot()
    status.success("🧠 Modelo de temas cargado ✔")

    time.sleep(0.5)
    status.info("🎭 Cargando generador de chistes…")
    joker = load_joke_model()
    status.success("😂 Generador de chistes cargado ✔")

    st.subheader("🚀 Procesando y generando humor…")
    progress_bar = st.progress(0)
    logs = st.empty()

    topics = [
        "politics", "celebrities", "sports", "animals",
        "technology", "health", "science", "relationships"
    ]
    total = len(df)

    for i, row in df.iterrows():
        if pd.notna(row["topic"]):
            continue

        text = str(row.get("headline", ""))
        if not text.strip():
            continue

        try:
            # 🔹 Zero-Shot Classification
            res = classifier(text, topics)
            df.at[i, "topic"] = res["labels"][0]
            df.at[i, "score"] = float(res["scores"][0])

            # 🔹 Joke Generation
            joke = joker(
                text[:100] + " 😆",
                max_length=60,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.9
            )[0]["generated_text"]

            df.at[i, "joke"] = joke.replace("\n", " ")

            # 🔹 Guardado incremental
            df.iloc[: i + 1].to_csv("progress_partial.csv", index=False)

            progress_bar.progress((i + 1) / total)
            logs.info(f"Procesado {i+1}/{total}")
            time.sleep(0.05)

        except Exception as e:
            logs.error(f"Error: {str(e)}")
            df.iloc[: i + 1].to_csv("progress_partial.csv", index=False)
            break

    status.success("✨ Procesamiento terminado ✔")

    # ======= Resultados =======
    st.subheader("📥 Descargar resultados")
    df.to_csv("final_results.csv", index=False)
    st.download_button(
        "📄 Descargar CSV completo",
        data=open("final_results.csv", "rb").read(),
        file_name="final_results.csv"
    )

    # ======= Gráfica =======
    st.subheader("📊 Distribución de temas")
    if df["topic"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y="topic", data=df[df["topic"].notna()], ax=ax,
                      order=df["topic"].value_counts().index)
        st.pyplot(fig)


st.markdown('<div class="footer">Designed by <b>Adolfo Camacho</b> 💻✨</div>',
            unsafe_allow_html=True)
