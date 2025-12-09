import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
import time
import os
import re

st.set_page_config(page_title="Humor Topic Classifier", layout="wide", page_icon="😂")

# ==========================================
# ✨ STYLES
# ==========================================
st.markdown("""
<style>
h1 { text-align:center; font-weight:900; font-size:32px; color:#00ADB5; }
footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>😂 Humor Topic Classifier</h1>", unsafe_allow_html=True)

# ==========================================
# 📂 CARGA DE ARCHIVO
# ==========================================
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV/TSV del Task-A", type=["csv","tsv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.subheader("📊 Vista previa de los datos")
    st.dataframe(df.head())

    # ==========================================
    # 🔍 TEXTO USABLE
    # ==========================================
    def build_text(row):
        if "headline" in df.columns and isinstance(row["headline"], str) and row["headline"] != "-":
            return row["headline"].strip()
        if "word1" in df.columns and "word2" in df.columns:
            return f"{str(row['word1'])} {str(row['word2'])}".strip()
        return ""

    df["text_clean"] = df.apply(build_text, axis=1)

    # ==========================================
    # 🧠 MODELO ZERO-SHOT (TEMAS)
    # ==========================================
    st.subheader("🧠 Cargando modelo Zero-Shot BERT…")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    st.success("Modelo de temas cargado ✔")

    topics_list = [
        "politics","celebrities","technology","animals","food",
        "sports","sex","crime","religion","health",
        "work","money","education","family","environment",
        "science","music","movies","internet","military"
    ]

    # ==========================================
    # 🤣 GENERADOR DE CHISTES (GPT-2)
    # ==========================================
    st.subheader("🎭 Cargando generador de chistes…")
    joke_gen = pipeline(
        "text-generation",
        model="gpt2",
        pad_token_id=50256,
        device=0 if torch.cuda.is_available() else -1
    )
    st.success("Listo para generar chistes 😂")

    def clean_joke(j):
        j = re.sub(r"\s+", " ", j)
        return j[:140]

    def generate_joke(txt, topic):
        prompt = f"Write a short funny joke about {topic}: {txt}. Joke:"
        out = joke_gen(prompt, max_length=60, temperature=0.95, num_return_sequences=1)
        joke = out[0]["generated_text"].split("Joke:")[-1]
        return clean_joke(joke)

    texts = df["text_clean"].tolist()
    total = len(texts)
    batch_size = 16  # más pequeño para ir más fluido

    topics, scores, jokes = [], [], []

    progress_bar = st.progress(0)
    status = st.empty()
    logs = st.container()
    start = time.time()

    output_file = "progress_partial.csv"

    # ==========================================
    # 🚀 CLASIFICAR + GENERAR CHISTES
    # ==========================================
    st.subheader(f"🔄 Clasificando {total} textos y generando chistes…")

    try:
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]

            # ---- clasificación de temas
            results = classifier(
                batch_texts,
                topics_list,
                hypothesis_template="This is about {}."
            )

            for r in results:
                topics.append(r["labels"][0])
                scores.append(float(r["scores"][0]))

            # ---- generación de chistes (uno por texto del batch)
            for idx, txt in enumerate(batch_texts):
                jokes.append(generate_joke(txt, topics[i+idx]))

            # Guardar progreso parcial en el DataFrame
            df.loc[:len(topics)-1, "topic"] = topics
            df.loc[:len(scores)-1, "score"] = scores
            df.loc[:len(jokes)-1, "joke"] = jokes

            df.to_csv(output_file, index=False)

            # Progreso
            prog = (i + batch_size) / total
            elapsed = time.time() - start
            eta = (elapsed/prog) - elapsed if prog > 0 else 0

            progress_bar.progress(min(prog, 1.0))
            status.info(f"✔ {min(i+batch_size,total)}/{total} • {prog*100:.1f}% • ⏱ {elapsed/60:.1f}m • ETA {eta/60:.1f}m")

            with logs:
                st.write(f"🟦 Batch procesado → filas hasta: {min(i+batch_size,total)}")

        status.success("🎉 Clasificación y generación de chistes completadas")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.warning("Se guardó el progreso parcial en progress_partial.csv")

    # ==========================================
    # 📈 DISTRIBUCIÓN DE TEMAS
    # ==========================================
    st.subheader("📈 Distribución de temas")
    if "topic" in df.columns and df["topic"].notna().any():
        fig, ax = plt.subplots(figsize=(10,6))
        sns.countplot(
            data=df[df["topic"].notna()],
            y="topic",
            order=df["topic"].value_counts().index,
            ax=ax
        )
        st.pyplot(fig)
    else:
        st.info("Aún no hay temas suficientes para graficar.")

    # ==========================================
    # 🎤 SECCIÓN E: “STAND-UP” POR TEMA
    # ==========================================
    st.subheader("🎤 Stand-up por tema")

    if "topic" in df.columns and "joke" in df.columns and df["topic"].notna().any():
        available_topics = sorted(df["topic"].dropna().unique().tolist())
        selected_topic = st.selectbox("Elige un tema para ver los chistes:", available_topics)

        n_show = st.slider("¿Cuántos chistes quieres ver?", min_value=3, max_value=50, value=10, step=1)

        topic_df = df[(df["topic"] == selected_topic) & df["joke"].notna()]

        if len(topic_df) == 0:
            st.info("No hay chistes generados para este tema todavía.")
        else:
            # mezclar para que no siempre sean los mismos
            topic_sample = topic_df.sample(min(n_show, len(topic_df)))

            st.markdown(f"### 🎭 Chistes del tema: **{selected_topic}**")
            for idx, row in topic_sample.iterrows():
                original = row.get("text_clean", "")
                joke = row.get("joke", "")
                st.markdown(
                    f"""
                    <div style="border-radius:10px; padding:10px 15px; margin-bottom:8px; background-color:#1F2933;">
                        <div style="color:#9CA3AF; font-size:12px; margin-bottom:4px;">
                            📝 <b>Texto original:</b> {original}
                        </div>
                        <div style="color:#F9FAFB; font-size:14px;">
                            😂 <b>Chiste:</b> {joke}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("Primero hay que terminar la clasificación y generación de chistes para ver esta sección.")

    # ==========================================
    # 📥 DESCARGA FINAL
    # ==========================================
    st.subheader("📦 Descargar resultados finales")
    st.download_button(
        "📥 Descargar CSV con temas y chistes",
        df.to_csv(index=False).encode("utf-8"),
        "classified_humor_with_jokes.csv"
    )
