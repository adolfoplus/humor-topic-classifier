import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

st.set_page_config(page_title="Humor Topic Classifier", layout="wide")

# ================================
# 👨‍💻 Estilo Terminal Linux
# ================================
terminal_style = """
<style>
.stApp {
    background-color: #000000;
}
html, body, [class*="st-"], p, label, span, h1, h2, h3, h4 {
    color: #00FF00 !important;
    font-family: "Courier New", monospace;
}
input, textarea, .stTextInput, .stFileUploader {
    background-color: #111 !important;
    color: #00FF00 !important;
    border: 1px solid #00FF00 !important;
}
div.stButton > button {
    background-color: #003300 !important;
    color: #00FF00 !important;
    border: 1px solid #00FF00;
    font-weight: bold;
}
.stProgress > div > div {
    background-color: #00FF00 !important;
}
table, th, td {
    color: #00FF00 !important;
    background-color: #000000 !important;
    border: 1px solid #00FF00 !important;
}
svg {
    background-color: #000000 !important;
}
footer, .stFooter {
    color: #00FF00 !important;
}
</style>
"""
st.markdown(terminal_style, unsafe_allow_html=True)

# ================================
# TITLE
# ================================
st.title("😄 Humor Topic Classifier — Task-A Zero-Shot")

uploaded_files = st.file_uploader(
    "📂 Carga tus archivos (.tsv)",
    type=["tsv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 Carga tus datasets para comenzar")
    st.stop()

# ================================
# LOAD DATA
# ================================
dfs = {}
for f in uploaded_files:
    lang = f.name.split("-")[-1].split(".")[0]
    df = pd.read_csv(f, sep="\t")
    df["lang"] = lang
    dfs[lang] = df

df_all = pd.concat(dfs.values(), ignore_index=True)
st.success(f"📊 Total de filas: {len(df_all)}")

def clean_text(row):
    if isinstance(row.get("headline"), str) and row["headline"].strip():
        return row["headline"].strip()
    if "word1" in row and "word2" in row:
        return f"{str(row['word1']).strip()} {str(row['word2']).strip()}"
    return ""

df_all["text_clean"] = df_all.apply(clean_text, axis=1).fillna("")

# ================================
# MODEL LOADING
# ================================
st.warning("⚙️ Cargando modelo...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

candidate_labels = [
    "política", "celebridades", "tecnología", "animales",
    "comida", "deportes", "sexo", "crimen",
    "religión", "salud", "trabajo", "dinero",
    "educación", "familia", "medio ambiente",
    "ciencia", "música", "cine", "internet", "militar"
]

texts = df_all["text_clean"].tolist()
batch_size = 8
topics, scores = [], []
output_filename = "clasificacion_BERT_parcial.csv"
cols = ["lang", "headline", "word1", "word2", "text_clean", "topic_bert", "topic_score"]

progress_bar = st.progress(0)
status_text = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()
download_placeholder = st.empty()

# ================================
# CHARTS
# ================================
def draw_charts(df):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    try:
        df["topic_bert"].value_counts().plot.barh(ax=axs[0], title="📌 Distribución por Tema", color="#00FF00")
        df.groupby("topic_bert")["topic_score"].mean().sort_values().plot.barh(
            ax=axs[1], title="📈 Score Promedio", color="#00FF00"
        )
    except: pass
    plt.tight_layout()
    return fig

# ================================
# CLASSIFICATION LOOP
# ================================
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]

    results = classifier(
        batch_texts,
        candidate_labels,
        hypothesis_template="This text is about {}."
    )

    for r in results:
        topics.append(r["labels"][0])
        scores.append(float(r["scores"][0]))

    df_all.loc[:len(topics)-1, "topic_bert"] = topics
    df_all.loc[:len(scores)-1, "topic_score"] = scores

    df_all[cols].to_csv(output_filename, index=False, encoding="utf-8-sig")

    progress_bar.progress(len(topics)/len(texts))
    status_text.write(f"> Procesadas: {len(topics)}/{len(texts)}")
    chart_placeholder.pyplot(draw_charts(df_all))
    table_placeholder.dataframe(df_all.head(10))

# ================================
# FINAL DOWNLOAD
# ================================
st.balloons()
st.success("✔ Clasificación Finalizada")

download_placeholder.download_button(
    label="⬇ Descargar Resultados",
    data=open(output_filename, "rb"),
    file_name="clasificacion_BERT_parcial.csv",
    mime="text/csv",
    key="download_partial"
)

chart_placeholder.pyplot(draw_charts(df_all))
table_placeholder.dataframe(df_all.head(20))
