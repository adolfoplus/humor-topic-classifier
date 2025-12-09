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
st.caption("Zero-shot + humor con estilo mexicano (ligero, sin pasarse)")

# =========================
# UPLOAD DE ARCHIVO
# =========================
uploaded_file = st.file_uploader(
    "📂 Sube tu archivo CSV/TSV del Task-A",
    type=["csv", "tsv"]
)

# =========================
# DEFINICIONES DE TEMAS
# =========================

TOPIC_INFO = {
    "politics": {
        "es": "política",
        "style": "Haz un chiste corto y gracioso con humor mexicano sobre política. Un poco sarcástico pero sin groserías fuertes."
    },
    "celebrities": {
        "es": "famosos y celebridades",
        "style": "Haz un chiste chismoso y divertido sobre celebridades, como si lo contara un amigo en la mesa."
    },
    "sports": {
        "es": "deportes",
        "style": "Haz un chiste futbolero o de deportes con humor mexicano, tipo plática de compas."
    },
    "animals": {
        "es": "animales",
        "style": "Haz un chiste tierno y simpático sobre animales, con un toque de humor mexicano."
    },
    "technology": {
        "es": "tecnología",
        "style": "Haz un chiste nerd de tecnología con humor mexicano, como si un compa techie se quejara riéndose."
    },
    "health": {
        "es": "salud",
        "style": "Haz un chiste ligero sobre salud o médicos, con humor mexicano y sin faltarle el respeto a nadie."
    },
    "food": {
        "es": "comida",
        "style": "Haz un chiste sabroso sobre comida, antojitos o tacos, con humor mexicano."
    },
    "religion": {
        "es": "religión",
        "style": "Haz un chiste muy suave y respetuoso sobre religión, con humor mexicano pero sin ofender."
    },
    "crime": {
        "es": "crimen",
        "style": "Haz un chiste ligero, tipo humor negro suave, sobre crimen, sin glorificar la violencia."
    },
    "money": {
        "es": "dinero",
        "style": "Haz un chiste sobre dinero, estar roto o pagar cuentas, con humor mexicano."
    },
    "work": {
        "es": "trabajo",
        "style": "Haz un chiste de oficina o trabajo godín con humor mexicano."
    },
    "family": {
        "es": "familia",
        "style": "Haz un chiste sobre la familia mexicana, con cariño y sin groserías fuertes."
    },
    "internet": {
        "es": "internet y redes sociales",
        "style": "Haz un chiste sobre memes, redes sociales o internet con humor mexicano."
    }
}

TOPIC_LABELS_EN = list(TOPIC_INFO.keys())


def safe_score_to_float(s):
    """Convierte la score de HF a float, tolerante a tensores."""
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
    """Construye el texto base del chiste: headline o word1+word2."""
    if "headline" in cols and isinstance(row.get("headline", ""), str) and row["headline"].strip() != "-":
        return row["headline"].strip()
    if "word1" in cols and "word2" in cols:
        return f"{str(row['word1'])} {str(row['word2'])}".strip()
    return ""


def clean_joke(joke: str, max_len: int = 140) -> str:
    """Limpia el chiste y lo recorta si es demasiado largo."""
    joke = joke.replace("<|endoftext|>", " ")
    joke = " ".join(joke.split())
    if len(joke) > max_len:
        joke = joke[:max_len-3] + "..."
    return joke.strip()


def make_prompt(text: str, topic_en: str) -> str:
    """Genera el prompt en español con estilo mexicano, sin que se vea en la salida."""
    info = TOPIC_INFO.get(topic_en, {
        "es": "algo",
        "style": "Haz un chiste corto y gracioso con humor mexicano."
    })
    topic_es = info["es"]
    style = info["style"]

    # Importante: terminar en "Chiste:" para poder cortar luego.
    prompt = (
        f"Título de noticia: \"{text}\".\n"
        f"{style}\n"
        f"Chiste:"
    )
    return prompt, topic_es


if uploaded_file:
    # =========================
    # CARGA DE DATA
    # =========================
    ext = uploaded_file.name.split(".")[-1].lower()
    sep = "," if ext == "csv" else "\t"
    df = pd.read_csv(uploaded_file, sep=sep)

    st.subheader("👀 Vista previa del archivo")
    st.dataframe(df.head(), use_container_width=True)

    cols = df.columns

    df["text_clean"] = df.apply(lambda r: build_text(r, cols), axis=1)
    total_rows = len(df)

    # =========================
    # CARGA DE MODELOS
    # =========================
    st.subheader("🧠 Cargando modelos de IA...")
    load_status = st.empty()
    load_status.info("🔍 Cargando modelo Zero-Shot para temas...")
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=0 if torch.cuda.is_available() else -1
    )
    load_status.success("🧠 Modelo de temas cargado ✔")

    time.sleep(0.4)
    load_status.info("🎭 Cargando modelo generador de chistes en español...")
    joke_model = pipeline(
        "text-generation",
        model="datificate/gpt2-spanish",
        device=0 if torch.cuda.is_available() else -1
    )
    load_status.success("😂 Generador de chistes en español cargado ✔")
    time.sleep(0.4)

    # =========================
    # PROCESAMIENTO
    # =========================
    st.subheader("🚀 Clasificando temas y generando chistes...")

    progress_bar = st.progress(0)
    status_text = st.empty()
    table_preview = st.empty()
    download_partial_slot = st.empty()

    topics_es = []
    topics_en_used = []
    scores = []
    jokes = []

    for idx, row in df.iterrows():
        text = row["text_clean"]
        if not isinstance(text, str) or not text.strip():
            topics_es.append(None)
            topics_en_used.append(None)
            scores.append(0.0)
            jokes.append("")
        else:
            # ---- Clasificación Zero-Shot ----
            res = classifier(
                text,
                TOPIC_LABELS_EN,
                hypothesis_template="This is about {}."
            )
            topic_en = res["labels"][0]
            score = safe_score_to_float(res["scores"][0])

            prompt, topic_es = make_prompt(text, topic_en)

            # ---- Generación del chiste en español ----
            out = joke_model(
                prompt,
                max_length=80,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.92,
                temperature=0.9
            )
            full = out[0]["generated_text"]

            # Cortar el prompt para que no se vea en el chiste
            generated_part = full[len(prompt):].strip()
            joke = clean_joke(generated_part)

            topics_es.append(topic_es)
            topics_en_used.append(topic_en)
            scores.append(score)
            jokes.append(joke)

        # ---- Actualizar progreso y UI ----
        df_partial = df.copy()
        df_partial["topic_en"] = topics_en_used + [None] * (total_rows - len(topics_en_used))
        df_partial["topic"] = topics_es + [None] * (total_rows - len(topics_es))
        df_partial["score"] = scores + [None] * (total_rows - len(scores))
        df_partial["joke"] = jokes + [""] * (total_rows - len(jokes))

        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.write(f"Procesado {idx+1}/{total_rows} filas ({progress*100:.1f}%)")

        # Mostrar solo último chunk
        cols_to_show = ["text_clean", "topic", "score", "joke"]
        preview_cols = [c for c in cols_to_show if c in df_partial.columns]
        table_preview.dataframe(df_partial[preview_cols].tail(20), use_container_width=True)

        # Botón de descarga parcial
        csv_partial = df_partial.to_csv(index=False).encode("utf-8")
        download_partial_slot.download_button(
            label=f"📥 Descargar progreso parcial ({idx+1}/{total_rows})",
            data=csv_partial,
            file_name="progreso_parcial_chistes.csv",
            mime="text/csv",
            key=f"partial_{idx}"
        )

    st.success("🎉 Proceso completado: temas asignados y chistes generados")

    # =========================
    # RESULTADOS FINALES
    # =========================
    df_final = df.copy()
    df_final["topic_en"] = topics_en_used
    df_final["topic"] = topics_es
    df_final["score"] = scores
    df_final["joke"] = jokes

    st.subheader("📥 Descargar resultados finales")
    csv_final = df_final.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📦 Descargar CSV final con temas y chistes",
        data=csv_final,
        file_name="humor_topics_with_jokes_es.csv",
        mime="text/csv"
    )

    # =========================
    # GRÁFICO DE TEMAS
    # =========================
    st.subheader("📊 Distribución de temas (en español)")
    if df_final["topic"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=df_final[df_final["topic"].notna()],
            y="topic",
            order=df_final["topic"].value_counts().index,
            ax=ax
        )
        ax.set_xlabel("Cantidad")
        ax.set_ylabel("Tema")
        st.pyplot(fig)
    else:
        st.info("Aún no hay suficientes temas para mostrar un gráfico.")
else:
    st.info("👆 Sube un archivo para comenzar.")
