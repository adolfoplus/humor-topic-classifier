import os
import time
import re

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Humor Topic Classifier",
    page_icon="😂",
    layout="wide"
)

# ==========================================
# ESTILOS
# ==========================================
st.markdown("""
<style>
h1 {
    text-align:center;
    font-weight:900;
    font-size:32px;
    color:#00ADB5;
}
footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>😂 Humor Topic Classifier</h1>", unsafe_allow_html=True)

# ==========================================
# PARÁMETROS GLOBALES
# ==========================================
# Forzamos CPU para evitar problemas de meta tensors
DEVICE = -1

TOPIC_LABELS = [
    "politics","celebrities","technology","animals","food",
    "sports","sex","crime","religion","health",
    "work","money","education","family","environment",
    "science","music","movies","internet","military"
]

# Estilos de humor por tema (mejor calidad de chistes)
TOPIC_STYLES = {
    "politics":   "Make a clever political satire, a bit sarcastic but light",
    "celebrities":"Make a playful joke about celebrity culture",
    "technology":"Make a nerdy tech joke",
    "animals":   "Make a cute, wholesome animal joke",
    "food":      "Make a tasty, light-hearted food joke",
    "sports":    "Make a competitive but friendly sports joke",
    "sex":       "Make a very light double-entendre, but keep it PG-13",
    "crime":     "Make a darkly comic but non-offensive crime joke",
    "religion":  "Make a very gentle, respectful, slightly ironic joke",
    "health":    "Make a relatable health or wellness joke",
    "work":      "Make an office / work-life joke",
    "money":     "Make a humorous joke about money and finances",
    "education": "Make a school / university joke",
    "family":    "Make a family-life joke",
    "environment":"Make an eco / climate joke, a bit ironic",
    "science":   "Make a science / physics nerd joke",
    "music":     "Make a music / band / pop-culture joke",
    "movies":    "Make a movie / cinema joke",
    "internet":  "Make an internet / meme / social media joke",
    "military":  "Make a light, respectful military joke"
}

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def build_text(row, df):
    if "headline" in df.columns and isinstance(row.get("headline", ""), str) and row["headline"] != "-":
        return row["headline"].strip()
    if "word1" in df.columns and "word2" in df.columns:
        return f"{str(row['word1'])} {str(row['word2'])}".strip()
    return ""

def clean_joke(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 140:
        text = text[:137] + "..."
    return text

def make_joke_prompt(text: str, topic: str) -> str:
    style = TOPIC_STYLES.get(topic, "Make a witty, surprising joke")
    prompt = (
        f"{style}. "
        f"Write a very short joke (max 2 sentences). "
        f"Topic: {topic}. Premise: {text}. "
        f"Punchline:"
    )
    return prompt

# ==========================================
# CARGA DE ARCHIVO
# ==========================================
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV/TSV del Task-A", type=["csv", "tsv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, sep=None, engine="python")

    st.subheader("📊 Vista previa")
    st.dataframe(df.head())

    # Texto limpio base
    df["text_clean"] = df.apply(lambda r: build_text(r, df), axis=1)
    texts = df["text_clean"].tolist()
    total = len(texts)

    # ======================================
    # OPCIÓN: REANUDAR DE CSV PARCIAL
    # ======================================
    resume = st.checkbox("🔁 Reanudar desde progress_partial.csv (si existe)")

    completed = 0
    topics, scores, jokes = [], [], []

    if resume and os.path.exists("progress_partial.csv"):
        try:
            prev = pd.read_csv("progress_partial.csv")
            # Transferimos columnas que existan
            for col in ["topic", "score", "joke"]:
                if col in prev.columns:
                    df[col] = prev[col]
            if "topic" in df.columns:
                completed = int(df["topic"].notna().sum())
            else:
                completed = 0

            if completed > 0:
                topics = df.loc[:completed-1, "topic"].tolist()
                scores = df.loc[:completed-1, "score"].tolist() if "score" in df.columns else []
                jokes  = df.loc[:completed-1, "joke"].tolist()  if "joke"  in df.columns else []

            st.info(f"🔁 Reanudando desde fila {completed} (de {total})")
        except Exception as e:
            st.warning(f"No se pudo reanudar: {e}")
            completed = 0
            topics, scores, jokes = [], [], []

    # ==========================================
    # MODELOS
    # ==========================================
    st.subheader("🧠 Cargando modelo Zero-Shot (rápido)…")
    # Modelo más ligero que bart-large-mnli
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device=DEVICE
    )
    st.success("Modelo de temas cargado ✔")

    st.subheader("🎭 Cargando generador de chistes…")
    # mantenemos gpt2 pero con mejores prompts
    joke_gen = pipeline(
        "text-generation",
        model="gpt2",
        pad_token_id=50256,
        device=DEVICE
    )
    st.success("Generador listo 😂")

    # ==========================================
    # PROCESAMIENTO
    # ==========================================
    batch_size = 32  # algo mayor para hacer menos llamadas
    st.subheader(f"🔄 Procesando {total} textos y generando humor…")

    visual_box   = st.empty()
    bar          = st.progress(0.0)
    progress_txt = st.empty()
    output_file  = "progress_partial.csv"

    start_time = time.time()

    try:
        for start in range(completed, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            # ---------- Estado: clasificando ----------
            frac = completed / total if total else 0.0
            bar.progress(frac)
            progress_txt.markdown(f"**Progreso:** {frac*100:.1f}% ({completed}/{total})")

            visual_box.markdown(
                f"""
                <div style="background:#1E293B;padding:18px;border-radius:10px;">
                    <p style="color:#38BDF8;"><b>🔍 Analizando temas…</b></p>
                    <p style="color:#94A3B8;">Textos {start+1} → {end}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            results = classifier(
                batch_texts,
                TOPIC_LABELS,
                hypothesis_template="This is about {}."
            )

            batch_topics = []
            batch_scores = []
            for r in results:
                batch_topics.append(r["labels"][0])
                s = r["scores"][0]
                # siempre lo forzamos a float, tolerante a tensores
                try:
                    if hasattr(s, "detach"):
                        s = s.detach()
                    if hasattr(s, "cpu"):
                        s = s.cpu()
                    if hasattr(s, "numpy"):
                        s = s.numpy()
                    s = float(s)
                except Exception:
                    s = float(s.item()) if hasattr(s, "item") else float(s)
                batch_scores.append(s)

            topics.extend(batch_topics)
            scores.extend(batch_scores)

            # ---------- Estado: generando chistes ----------
            visual_box.markdown(
                f"""
                <div style="background:#1E293B;padding:18px;border-radius:10px;">
                    <p style="color:#FACC15;"><b>✍️ Creando chistes con mejor estilo…</b></p>
                    <p style="color:#94A3B8;">Humor por tema (política, animales, internet, etc.)</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            batch_jokes = []
            for txt, topic in zip(batch_texts, batch_topics):
                prompt = make_joke_prompt(txt, topic)
                out = joke_gen(
                    prompt,
                    max_length=70,
                    temperature=0.9,
                    num_return_sequences=1,
                    do_sample=True
                )
                raw = out[0]["generated_text"].split("Punchline:")[-1]
                batch_jokes.append(clean_joke(raw))

            jokes.extend(batch_jokes)

            # ---------- Actualizar progreso global ----------
            completed = len(topics)

            # rellenar DataFrame
            df.loc[:completed-1, "topic"] = topics
            df.loc[:completed-1, "score"] = scores
            df.loc[:completed-1, "joke"]  = jokes

            # guardar parcial
            df.to_csv(output_file, index=False)

            frac = completed / total if total else 1.0
            elapsed = time.time() - start_time
            eta = (elapsed / frac - elapsed) if frac > 0 else 0

            visual_box.markdown(
                f"""
                <div style="background:#1E293B;padding:18px;border-radius:10px;">
                    <p style="color:#4ADE80;"><b>✨ Batch completado</b></p>
                    <p style="color:#94A3B8;">Avance total: {frac*100:.1f}% ({completed}/{total})</p>
                    <p style="color:#64748B;">Tiempo: {elapsed/60:.1f} min · ETA: {eta/60:.1f} min</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            bar.progress(frac)
            progress_txt.markdown(f"**Progreso:** {frac*100:.1f}% ({completed}/{total})")

        st.success("🎉 Proceso completado: temas + chistes generados")

    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {e}")
        st.warning("Progreso parcial guardado en progress_partial.csv")

    # ==========================================
    # GRÁFICO DE TEMAS
    # ==========================================
    st.subheader("📈 Distribución de temas")
    if "topic" in df.columns and df["topic"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            data=df[df["topic"].notna()],
            y="topic",
            order=df["topic"].value_counts().index,
            ax=ax
        )
        st.pyplot(fig)
    else:
        st.info("Aún no hay suficientes temas para graficar.")

    # ==========================================
    # STAND-UP MODE
    # ==========================================
    st.subheader("🎤 Stand-Up por tema")
    if "topic" in df.columns and "joke" in df.columns and df["topic"].notna().any():
        temas = sorted(df["topic"].dropna().unique().tolist())
        sel   = st.selectbox("Elige un tema", temas)
        n     = st.slider("Chistes a mostrar", 3, 50, 10)

        dff = df[(df["topic"] == sel) & df["joke"].notna()]
        if len(dff) == 0:
            st.info("No hay chistes generados para este tema aún.")
        else:
            sample = dff.sample(min(n, len(dff)))
            for _, row in sample.iterrows():
                st.markdown(
                    f"""
                    <div style="background:#0F172A;border-radius:10px;
                                padding:12px 15px;margin-bottom:8px;">
                        <p style="color:#9CA3AF;font-size:12px;">
                            📝 <b>Texto:</b> {row['text_clean']}
                        </p>
                        <p style="color:#F9FAFB;font-size:14px;">
                            🤣 <b>Chiste:</b> {row['joke']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ==========================================
    # DESCARGA FINAL
    # ==========================================
    st.subheader("📦 Descargar resultados")
    st.download_button(
        "📥 Descargar CSV con temas y chistes",
        df.to_csv(index=False).encode("utf-8"),
        "classified_humor_with_jokes.csv"
    )
