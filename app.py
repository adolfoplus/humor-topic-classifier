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
    page_title="Humor Hacker Console 😂",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# ESTILO TERMINAL HACKER
# =========================
st.markdown("""
<style>
body {
    background-color: #020b02;
}
.block-container {
    padding-top: 1.5rem;
}
* {
    font-family: "Source Code Pro", "Consolas", "Courier New", monospace;
}
h1, h2, h3, h4, h5, h6, p, span, li, label {
    color: #00ff7f !important;
}
a {
    color: #00e0ff !important;
}
table tbody tr:hover {
    background-color: rgba(0, 255, 127, 0.08) !important;
}
.stProgress > div > div {
    background-color: #00ff7f;
}
hr {
    border-top: 1px solid rgba(0,255,127,0.35);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div style="
    border: 1px solid rgba(0,255,127,0.4);
    padding: 0.75rem 1rem;
    background: radial-gradient(circle at top left, rgba(0,255,127,0.2), transparent 55%);
    box-shadow: 0 0 18px rgba(0,255,127,0.25);
">
    <div style="font-size: 1.6rem; color:#00ff7f;">
        [ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console
    </div>
    <div style="font-size: 0.95rem; color:#7CFC00; margin-top: 4px;">
        Zero-shot BART + Spanish GPT :: Generating jokes with Mexican flavor...
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# CRÉDITO PROFESIONAL AL INICIO
# =========================
st.markdown("""
<div style="width:100%; text-align:center; margin-top:10px; margin-bottom:18px;">
    <div style="font-size:14px; color:#00ff7f; text-shadow:0 0 6px rgba(0,255,127,0.7);">
        Designed by <strong>Adolfo Camacho</strong>
    </div>
    <div style="font-size:13px; margin-top:4px;">
        🔗 <a href="https://www.linkedin.com/in/adolfo-camacho-328a2a157" target="_blank">
        LinkedIn: adolfo-camacho-328a2a157</a>
    </div>
    <div style="font-size:13px; margin-top:2px;">
        📧 <a href="mailto:turboplay333@gmail.com">turboplay333@gmail.com</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# UPLOAD DE ARCHIVO
# =========================
st.markdown("`[ INPUT ]`  Load SemEval Task-A file (CSV / TSV)")
uploaded_file = st.file_uploader(
    "Drop or browse your CSV/TSV file here",
    type=["csv", "tsv"]
)

# =========================
# DEFINICIÓN DE TEMAS Y ESTILOS DE HUMOR
# =========================
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
        "style": "Haz un chiste de humor negro suave sobre crimen, sin glorificar la violencia."
    },
    "money": {
        "es": "dinero",
        "style": "Haz un chiste sobre estar roto y pagar cuentas, estilo mexicano."
    },
    "work": {
        "es": "trabajo",
        "style": "Haz un chiste godín de oficina, con humor mexicano."
    },
    "family": {
        "es": "familia",
        "style": "Haz un chiste sobre la familia mexicana, con cariño."
    },
    "internet": {
        "es": "internet y redes sociales",
        "style": "Haz un chiste de memes o redes sociales, con humor mexicano."
    }
}

TOPIC_LABELS_EN = list(TOPIC_INFO.keys())

# =========================
# FUNCIONES AUXILIARES
# =========================
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

# =========================
# LÓGICA PRINCIPAL
# =========================
if uploaded_file:
    # Detectar formato
    ext = uploaded_file.name.split(".")[-1].lower()
    sep = "," if ext == "csv" else "\t"
    df = pd.read_csv(uploaded_file, sep=sep)

    st.markdown("`[ PREVIEW ]`  First rows of loaded file")
    st.dataframe(df.head(), use_container_width=True)

    df["text_clean"] = df.apply(lambda r: build_text(r, df.columns), axis=1)
    total_rows = len(df)

    # =========================
    # CARGA DE MODELOS
    # =========================
    st.markdown("`[ MODELS ]`  Initializing Zero-Shot & Joke Generator...")
    load_txt = st.empty()

    device_id = 0 if torch.cuda.is_available() else -1
    device_label = "cuda:0" if torch.cuda.is_available() else "cpu"

    load_txt.markdown(f"`-> Zero-Shot on {device_label} ...`")
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=device_id
    )
    load_txt.markdown(f"`-> Zero-Shot online ✔  [{device_label}]`")
    time.sleep(0.4)

    load_txt.markdown("`-> Loading Spanish GPT joke model ...`")
    joke_model = pipeline(
        "text-generation",
        model="datificate/gpt2-spanish",
        device=device_id
    )
    load_txt.markdown("`-> Joke model online ✔`")
    time.sleep(0.4)

    st.markdown("---")
    st.markdown("`[ PROCESSING ]`  Classifying topics & generating jokes")

    # =========================
    # ANIMACIÓN H1 – MATRIX PULSE
    # =========================
    anim_slot = st.empty()
    bars_frames = [
        "█ ▇ ▆ ▅ ▄ ▂ ▁ ▂ ▄ ▅ ▆ ▇ █",
        "▇ ▆ ▅ ▄ ▂ ▁ ▂ ▄ ▅ ▆ ▇ █ ▇",
        "▆ ▅ ▄ ▂ ▁ ▂ ▄ ▅ ▆ ▇ █ ▇ ▆",
        "▅ ▄ ▂ ▁ ▂ ▄ ▅ ▆ ▇ █ ▇ ▆ ▅",
        "▄ ▂ ▁ ▂ ▄ ▅ ▆ ▇ █ ▇ ▆ ▅ ▄",
        "▂ ▁ ▂ ▄ ▅ ▆ ▇ █ ▇ ▆ ▅ ▄ ▂",
        "▁ ▂ ▄ ▅ ▆ ▇ █ ▇ ▆ ▅ ▄ ▂ ▁",
    ]
    frame_i = 0

    # Barra de progreso y vistas
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

        # Actualizar animación tipo terminal
        anim_slot.markdown(
            f"<div style='font-size:26px; color:#00ff7f; text-align:center; "
            f"text-shadow:0 0 10px rgba(0,255,127,0.8);'>{bars_frames[frame_i]}</div>",
            unsafe_allow_html=True
        )
        frame_i = (frame_i + 1) % len(bars_frames)

        if not isinstance(text, str) or not text.strip():
            topics_es.append(None)
            topics_en_used.append(None)
            scores.append(0.0)
            jokes.append("")
        else:
            # Clasificación de tema
            res = classifier(text, TOPIC_LABELS_EN, hypothesis_template="This is about {}.")
            topic_en = res["labels"][0]
            score = safe_score_to_float(res["scores"][0])

            # Prompt y chiste
            prompt, topic_es = make_prompt(text, topic_en)
            out = joke_model(
                prompt,
                max_length=80,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.92,
                temperature=0.9
            )
            full = out[0]["generated_text"]
            generated = full[len(prompt):].strip()
            joke = clean_joke(generated)

            topics_es.append(topic_es)
            topics_en_used.append(topic_en)
            scores.append(score)
            jokes.append(joke)

        # DataFrame parcial
        df_partial = df.copy()
        df_partial["topic_en"] = topics_en_used + [None] * (total_rows - len(topics_en_used))
        df_partial["topic"] = topics_es + [None] * (total_rows - len(topics_es))
        df_partial["score"] = scores + [None] * (total_rows - len(scores))
        df_partial["joke"] = jokes + [""] * (total_rows - len(jokes))

        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.markdown(
            f"`row {idx+1}/{total_rows}  ::  {progress*100:5.1f}% complete`"
        )

        preview_table.dataframe(
            df_partial[["text_clean", "topic", "score", "joke"]].tail(20),
            use_container_width=True
        )

        partial_download_btn.download_button(
            label=f"📥 Download partial CSV ({idx+1}/{total_rows})",
            data=df_partial.to_csv(index=False).encode("utf-8"),
            file_name="progress_partial_jokes.csv",
            mime="text/csv",
            key=f"partial_{idx}"
        )

    st.success("`[ DONE ]`  All rows processed. Topics + jokes ready ✔")

    # =========================
    # RESULTADOS FINALES
    # =========================
    df_final = df.copy()
    df_final["topic_en"] = topics_en_used
    df_final["topic"] = topics_es
    df_final["score"] = scores
    df_final["joke"] = jokes

    st.markdown("`[ OUTPUT ]`  Export classified humor dataset")
    st.download_button(
        label="📥 Download full CSV",
        data=df_final.to_csv(index=False).encode("utf-8"),
        file_name="humor_es_results_hacker_console.csv",
        mime="text/csv"
    )

    # =========================
    # GRÁFICO DE TEMAS
    # =========================
    st.markdown("`[ ANALYTICS ]`  Topic distribution (ES)")
    if df_final["topic"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(
            data=df_final[df_final["topic"].notna()],
            y="topic",
            order=df_final["topic"].value_counts().index,
            ax=ax,
            color="#00ff7f"
        )
        ax.set_xlabel("Count", color="#00ff7f")
        ax.set_ylabel("Topic (ES)", color="#00ff7f")
        for spine in ax.spines.values():
            spine.set_edgecolor("#00ff7f")
        ax.tick_params(colors="#00ff7f")
        fig.patch.set_facecolor("#020b02")
        ax.set_facecolor("#020b02")
        st.pyplot(fig)
    else:
        st.info("No hay suficientes datos para el gráfico.")

else:
    st.markdown("`[ IDLE ]`  Waiting for input file... drop CSV/TSV above.")
