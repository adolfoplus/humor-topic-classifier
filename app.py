import streamlit as st
import pandas as pd
import time
import torch
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Humor Topic Classifier 😂",
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# ESTILOS OSCUROS PREMIUM
# =========================
st.markdown("""
<style>
body {
    background-color: #0B1120;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3, h4, h5, h6, p, span, li {
    color: #E5E7EB;
}
table tbody tr:hover {
    background-color: #111827 !important;
}
.stProgress > div > div {
    background-color: #38BDF8;
}
</style>
""", unsafe_allow_html=True)

st.title("😂 Humor Topic Classifier + Generador de Chistes en Español")
st.caption("Zero-shot + humor con estilo mexicano (ligero y divertido)")

# =========================
# UPLOAD DE ARCHIVO
# =========================
uploaded_file = st.file_uploader(
    "📂 Sube tu archivo CSV/TSV del Task-A",
    type=["csv", "tsv"]
)

TOPIC_INFO = {
    "politics": {
        "es": "política",
        "style": "Haz un chiste corto y gracioso con humor mexicano sobre política. Sarcástico sin pasarse."
    },
    "celebrities": {
        "es": "celebridades",
        "style": "Haz un chiste divertido y chismoso sobre celebridades."
    },
    "sports": {
        "es": "deportes",
        "style": "Haz un chiste futbolero o de deportes tipo compas echando relajo."
    },
    "animals": {
        "es": "animales",
        "style": "Haz un chiste tierno y simpático sobre animales, con humor ligero."
    },
    "technology": {
        "es": "tecnología",
        "style": "Haz un chiste nerd de tecnología, como compa techie burlándose."
    },
    "health": {
        "es": "salud",
        "style": "Haz un chiste ligero sobre salud, sin faltar al respeto."
    },
    "food": {
        "es": "comida",
        "style": "Haz un chiste sabroso tipo taquiza, con humor mexicano."
    },
    "religion": {
        "es": "religión",
        "style": "Haz un chiste suave y respetuoso sobre religión."
    },
    "crime": {
        "es": "crimen",
        "style": "Humor negro suave sobre crimen, sin glorificar la violencia."
    },
    "money": {
        "es": "dinero",
        "style": "Haz un chiste sobre estar roto y pagar cuentas, estilo mexicano."
    },
    "work": {
        "es": "trabajo",
        "style": "Chistes godínez de oficina, con humor mexicano."
    },
    "family": {
        "es": "familia",
        "style": "Humor sobre la familia mexicana, con cariño."
    },
    "internet": {
        "es": "internet y redes sociales",
        "style": "Haz un chiste de memes o redes sociales, con humor mexicano."
    }
}

TOPIC_LABELS_EN = list(TOPIC_INFO.keys())


def safe_score_to_float(s):
    try:
        if hasattr(s, "detach"):
            s = s.detach()
        if hasattr(s, "cpu"):
            s = s.cpu()
        if hasattr(s, "numpy"):
            s = s.numpy()
        return float(s)
    except Exception:
        return float(s.item()) if hasattr(s, "item") else float(s)


def build_text(row, cols):
    if "headline" in cols and isinstance(row.get("headline", ""), str) and row["headline"].strip() != "-":
        return row["headline"].strip()
    if "word1" in cols and "word2" in cols:
        return f"{str(row['word1'])} {str(row['word2'])}".strip()
    return ""


def clean_joke(joke, max_len=140):
    joke = joke.replace("<|endoftext|>", " ")
    joke = " ".join(joke.split())
    if len(joke) > max_len:
        joke = joke[:max_len-3] + "..."
    return joke.strip()


def make_prompt(text, topic_en):
    info = TOPIC_INFO.get(topic_en, {"es": "algo", "style": "Haz un chiste gracioso en español."})
    prompt = (
        f"Título de noticia: \"{text}\".\n"
        f"{info['style']}\n"
        f"Chiste:"
    )
    return prompt, info["es"]


if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    sep = "," if ext == "csv" else "\t"
    df = pd.read_csv(uploaded_file, sep=sep)

    st.subheader("👀 Vista previa")
    st.dataframe(df.head(), use_container_width=True)

    df["text_clean"] = df.apply(lambda r: build_text(r, df.columns), axis=1)
    total_rows = len(df)

    # =========================
    # CARGA DE MODELOS
    # =========================
    st.subheader("🧠 Cargando modelos…")
    load_txt = st.empty()

    load_txt.info("Cargando Zero-Shot...")
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=0 if torch.cuda.is_available() else -1
    )
    load_txt.success("Modelo de temas listo ✔")
    time.sleep(0.5)

    load_txt.info("Cargando generador de chistes en español…")
    joke_model = pipeline(
        "text-generation",
        model="datificate/gpt2-spanish",
        device=0 if torch.cuda.is_available() else -1
    )
    load_txt.success("Generador de humor listo ✔")
    time.sleep(0.5)

    # =========================
    # PROCESAMIENTO
    # =========================
    st.subheader("🚀 Procesando y generando chistes…")

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_table = st.empty()
    partial_download_btn = st.empty()

    topics_es = []
    topics_en_used = []
    scores = []
    jokes = []

    for idx, row in df.iterrows():
        text = row["text_clean"]

        if not isinstance(text, str) or not text:
            topics_es.append(None)
            topics_en_used.append(None)
            scores.append(0.0)
            jokes.append("")
        else:
            res = classifier(text, TOPIC_LABELS_EN, hypothesis_template="This is about {}.")
            topic_en = res["labels"][0]
            score = safe_score_to_float(res["scores"][0])

            prompt, topic_es = make_prompt(text, topic_en)
            out = joke_model(prompt, max_length=80, num_return_sequences=1, do_sample=True, top_p=0.92, temperature=0.9)
            full = out[0]["generated_text"]
            generated = full[len(prompt):].strip()
            joke = clean_joke(generated)

            topics_es.append(topic_es)
            topics_en_used.append(topic_en)
            scores.append(score)
            jokes.append(joke)

        df_partial = df.copy()
        df_partial["topic_en"] = topics_en_used + [None]*(total_rows-len(topics_en_used))
        df_partial["topic"] = topics_es + [None]*(total_rows-len(topics_es))
        df_partial["score"] = scores + [None]*(total_rows-len(scores))
        df_partial["joke"] = jokes + [""]*(total_rows-len(jokes))

        progress = (idx+1)/total_rows
        progress_bar.progress(progress)
        status_text.write(f"Procesado {idx+1}/{total_rows} ({progress*100:.1f}%)")

        preview_table.dataframe(df_partial[["text_clean","topic","score","joke"]].tail(20), use_container_width=True)

        partial_download_btn.download_button(
            label=f"📥 Descargar progreso ({idx+1}/{total_rows})",
            data=df_partial.to_csv(index=False).encode("utf-8"),
            file_name="progreso_parcial_chistes.csv",
            mime="text/csv",
            key=f"partial_{idx}"
        )

    st.success("🎉 ¡Todo listo! Temas y chistes generados ✔")

    df_final = df.copy()
    df_final["topic_en"] = topics_en_used
    df_final["topic"] = topics_es
    df_final["score"] = scores
    df_final["joke"] = jokes

    st.subheader("📦 Descargar resultados finales")
    st.download_button(
        label="📥 Descargar CSV completo",
        data=df_final.to_csv(index=False).encode("utf-8"),
        file_name="humor_es_resultados.csv",
        mime="text/csv"
    )

    # =========================
    # GRÁFICO DE TEMAS
    # =========================
    st.subheader("📊 Distribución de temas")
    if df_final["topic"].notna().any():
        fig, ax = plt.subplots(figsize=(7,4))
        sns.countplot(data=df_final[df_final["topic"].notna()], y="topic", ax=ax)
        ax.set_xlabel("Cantidad")
        ax.set_ylabel("Tema")
        st.pyplot(fig)
    else:
        st.info("No hay suficientes datos para el gráfico.")

    # =========================
    # CRÉDITO PROFESIONAL
    # =========================
    st.markdown("""
    <div style="
        width:100%;
        text-align:center;
        margin-top:40px;
        padding:12px;
        ">
        <div style="
            font-size:16px;
            color:#38BDF8;
            text-shadow:0px 0px 8px rgba(56,189,248,0.8);
            margin-bottom:6px;">
            Designed by <strong>Adolfo Camacho</strong>
        </div>

        <div style="margin-bottom:4px;">
            🔗 <a href="https://www.linkedin.com/in/adolfo-camacho-328a2a157"
            target="_blank" style="color:#60A5FA;
            text-decoration:none;font-size:14px;">
            LinkedIn: Adolfo Camacho</a>
        </div>

        <div>
            📧 <a href="mailto:turboplay333@gmail.com"
            style="color:#34D399;text-decoration:none;font-size:14px;">
            turboplay333@gmail.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Sube un archivo para comenzar.")
