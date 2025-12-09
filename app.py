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
# 🎨 Estilos
# ==========================================
st.markdown("""
<style>
h1 { text-align:center; font-weight:900; font-size:32px; color:#00ADB5; }
footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>😂 Humor Topic Classifier</h1>", unsafe_allow_html=True)

# ==========================================
# 📂 Upload
# ==========================================
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV/TSV del Task-A", type=["csv","tsv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.subheader("📊 Vista previa")
    st.dataframe(df.head())

    # ==========================================
    # 🔍 Extraer texto usable
    # ==========================================
    def build_text(row):
        if "headline" in df.columns and isinstance(row["headline"], str) and row["headline"] != "-":
            return row["headline"].strip()
        if "word1" in df.columns and "word2" in df.columns:
            return f"{str(row['word1'])} {str(row['word2'])}".strip()
        return ""

    df["text_clean"] = df.apply(build_text, axis=1)

    texts = df["text_clean"].tolist()
    total = len(texts)
    batch_size = 16

    # ==========================================
    # 🧠 Zero-Shot BERT
    # ==========================================
    st.subheader("🧠 Cargando modelo Zero-Shot...")
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
    # 🤣 Generador de chistes GPT-2
    # ==========================================
    st.subheader("🎭 Cargando generador de chistes...")
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
        result = joke_gen(prompt, max_length=60, temperature=0.95, num_return_sequences=1)
        joke = result[0]["generated_text"].split("Joke:")[-1].strip()
        return clean_joke(joke)

    topics, scores, jokes = [], [], []
    progress_bar = st.progress(0)
    visual_box = st.empty()
    output_file = "progress_partial.csv"
    start = time.time()

    # ==========================================
    # 🚀 Procesamiento batch con visual reactivo
    # ==========================================
    st.subheader(f"🔄 Procesando {total} textos y generando humor…")

    try:
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]

            # 🔵 Estado: Detectando tema
            visual_box.markdown(
                f"""
                <div style="background:#1E293B; padding:18px; border-radius:10px;">
                    <p style="color:#38BDF8;"><b>🔍 Analizando tema...</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            results = classifier(
                batch_texts,
                topics_list,
                hypothesis_template="This is about {}."
            )
            for r in results:
                topics.append(r["labels"][0])
                scores.append(float(r["scores"][0]))

            # 🟡 Estado: Generando chiste
            visual_box.markdown(
                f"""
                <div style="background:#1E293B; padding:18px; border-radius:10px;">
                    <p style="color:#FACC15;"><b>✍️ Creando chiste...</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            for idx, txt in enumerate(batch_texts):
                jokes.append(generate_joke(txt, topics[i+idx]))

            # Guardar progreso parcial
            df.loc[:len(topics)-1, "topic"] = topics
            df.loc[:len(scores)-1, "score"] = scores
            df.loc[:len(jokes)-1, "joke"] = jokes
            df.to_csv(output_file, index=False)

            prog = min((i+batch_size)/total, 1.0)

            # 🟢 Estado: Batch completado
            visual_box.markdown(
                f"""
                <div style="background:#1E293B; padding:18px; border-radius:10px;">
                    <p style="color:#4ADE80;"><b>✨ Batch completado</b></p>
                    <p style="color:#CBD5E1;">Progreso: {prog*100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress_bar.progress(prog)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.warning("Progreso parcial guardado en progress_partial.csv")

    # ==========================================
    # 📈 Gráfico de temas
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

    # ==========================================
    # 🎤 Stand-Up Mode
    # ==========================================
    st.subheader("🎤 Stand-Up por tema")
    if "topic" in df.columns and "joke" in df.columns:
        available_topics = sorted(df["topic"].dropna().unique().tolist())
        selected = st.selectbox("Elige un tema:", available_topics)
        n = st.slider("¿Cuántos chistes?", 3, 50, 10)

        sample = df[df["topic"] == selected].sample(min(n, len(df[df["topic"] == selected])))

        for _, row in sample.iterrows():
            st.markdown(
                f"""
                <div style="background:#0f172a; border-radius:10px; padding:12px 15px; margin-bottom:8px;">
                    <p style="color:#9CA3AF; font-size:12px;">📝 <b>Texto:</b> {row['text_clean']}</p>
                    <p style="color:#F9FAFB; font-size:14px;">🤣 <b>Chiste:</b> {row['joke']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ==========================================
    # 📥 DESCARGA FINAL
    # ==========================================
    st.subheader("📦 Descargar resultados")
    st.download_button(
        "📥 Descargar CSV con temas y chistes",
        df.to_csv(index=False).encode("utf-8"),
        "classified_humor_with_jokes.csv"
    )
