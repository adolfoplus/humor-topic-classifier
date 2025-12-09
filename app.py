import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
import time
import re

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Humor Topic Classifier",
    page_icon="😂",
    layout="wide"
)

# ==========================================
# STYLE
# ==========================================
st.markdown("""
<style>
h1 {
    text-align: center;
    font-weight: 900;
    font-size: 32px;
    color: #00ADB5;
}
footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>😂 Humor Topic Classifier</h1>", unsafe_allow_html=True)

# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV/TSV del Task-A", type=["csv","tsv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.subheader("📊 Vista previa")
    st.dataframe(df.head())

    # Construir texto
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
    # LOAD ZERO-SHOT (TOPICS)
    # ==========================================
    st.subheader("🧠 Cargando modelo Zero-Shot…")
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
    # LOAD JOKE GENERATOR
    # ==========================================
    st.subheader("🎭 Cargando generador de chistes…")
    joke_gen = pipeline(
        "text-generation",
        model="gpt2",
        pad_token_id=50256,
        device=0 if torch.cuda.is_available() else -1
    )
    st.success("Generador listo 😂")

    def clean_joke(j):
        j = re.sub(r"\s+"," ",j)
        return j[:140]

    def generate_joke(txt, topic):
        prompt = f"Write a short funny joke about {topic}: {txt}. Joke:"
        out = joke_gen(prompt, max_length=60, temperature=0.95)
        joke = out[0]["generated_text"].split("Joke:")[-1].strip()
        return clean_joke(joke)

    topics, scores, jokes = [], [], []

    # ==========================================
    # PROGRESS UI
    # ==========================================
    st.subheader(f"🔄 Procesando {total} textos y generando humor…")
    visual_box = st.empty()
    bar = st.progress(0)
    progress_text = st.empty()

    output_file = "progress_partial.csv"
    completed = 0
    start = time.time()

    try:
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]

            # Estado → clasificando
            frac = completed / total
            bar.progress(frac)
            progress_text.markdown(f"**Progreso:** {frac*100:.1f}% ({completed}/{total})")

            visual_box.markdown(
                f"""
                <div style="background:#1E293B;padding:18px;border-radius:10px;">
                <p style="color:#38BDF8;"><b>🔍 Analizando temas…</b></p>
                <p style="color:#94A3B8;">Textos {completed+1} → {completed+len(batch_texts)}</p>
                </div>
                """,unsafe_allow_html=True
            )

            results = classifier(
                batch_texts,
                topics_list,
                hypothesis_template="This is about {}."
            )

            batch_topics, batch_scores = [], []

            for r in results:
                batch_topics.append(r["labels"][0])
                s = r["scores"][0]
                if hasattr(s,"detach"):
                    s = s.detach().cpu().numpy()
                batch_scores.append(float(s))

            topics.extend(batch_topics)
            scores.extend(batch_scores)

            # Estado → generando chistes
            visual_box.markdown(
                f"""
                <div style="background:#1E293B;padding:18px;border-radius:10px;">
                <p style="color:#FACC15;"><b>✍️ Creando chistes…</b></p>
                <p style="color:#94A3B8;">El humor toma su tiempo 😄</p>
                </div>
                """,unsafe_allow_html=True
            )

            batch_jokes=[]
            for txt,topic in zip(batch_texts,batch_topics):
                batch_jokes.append(generate_joke(txt,topic))
            jokes.extend(batch_jokes)

            # Actualizar estado global
            completed = len(topics)

            df.loc[:completed-1,"topic"]=topics
            df.loc[:completed-1,"score"]=scores
            df.loc[:completed-1,"joke"]=jokes
            df.to_csv(output_file,index=False)

            frac = completed/total
            elapsed = time.time()-start
            eta = (elapsed/frac - elapsed) if frac>0 else 0

            visual_box.markdown(
                f"""
                <div style="background:#1E293B;padding:18px;border-radius:10px;">
                <p style="color:#4ADE80;"><b>✨ Batch completado</b></p>
                <p style="color:#94A3B8;">{frac*100:.1f}% ({completed}/{total})</p>
                <p style="color:#64748B;">ETA: {eta/60:.1f} min</p>
                </div>
                """,unsafe_allow_html=True
            )
            bar.progress(frac)
            progress_text.markdown(f"**Progreso:** {frac*100:.1f}% ({completed}/{total})")

        st.success("🎉 ¡Proceso completado!")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.warning("Progreso parcial guardado en progress_partial.csv")

    # ==========================================
    # GRAFICO DE TEMAS
    # ==========================================
    st.subheader("📈 Distribución de temas")
    if "topic" in df.columns and df["topic"].notna().any():
        fig, ax = plt.subplots(figsize=(10,6))
        sns.countplot(data=df[df["topic"].notna()],y="topic",
                      order=df["topic"].value_counts().index,ax=ax)
        st.pyplot(fig)

    # ==========================================
    # STAND-UP MODE
    # ==========================================
    st.subheader("🎤 Stand-Up por tema")
    if "topic" in df.columns and "joke" in df.columns:
        temas = sorted(df["topic"].dropna().unique().tolist())
        sel = st.selectbox("Elige un tema", temas)
        n = st.slider("Chistes a mostrar", 3, 50, 10)
        dff = df[df["topic"]==sel].dropna(subset=["joke"])
        if len(dff)>0:
            sample = dff.sample(min(n,len(dff)))
            for _,row in sample.iterrows():
                st.markdown(
                    f"""
                    <div style="background:#0F172A;border-radius:10px;
                    padding:12px 15px;margin-bottom:8px;">
                    <p style="color:#9CA3AF;font-size:12px;">📝 {row['text_clean']}</p>
                    <p style="color:#F9FAFB;font-size:15px;">🤣 {row['joke']}</p>
                    </div>
                    """,unsafe_allow_html=True
                )

    # ==========================================
    # DOWNLOAD FINAL
    # ==========================================
    st.subheader("📦 Descargar resultados")
    st.download_button(
        "📥 Descargar CSV con temas y chistes",
        df.to_csv(index=False).encode("utf-8"),
        "classified_humor_with_jokes.csv"
    )
