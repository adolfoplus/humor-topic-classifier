import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
from tqdm.auto import tqdm

st.set_page_config(page_title="Clasificador Task-A", layout="wide")

st.title("😄 Clasificador de Temas para Task-A (Zero-Shot + Batches)")
st.write("Sube tus archivos TSV de la competencia (EN / ES / ZH)")

uploaded_files = st.file_uploader(
    "📂 Carga tus archivos Task-A:",
    type=["tsv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 Carga al menos un archivo .tsv para comenzar")
    st.stop()

# ================================
# LEER Y UNIR DATASETS
# ================================
dfs = {}
for f in uploaded_files:
    lang = f.name.split("-")[-1].split(".")[0]  # en / es / zh
    df = pd.read_csv(f, sep="\t")
    df["lang"] = lang
    dfs[lang] = df

df_all = pd.concat(dfs.values(), ignore_index=True)

st.success(f"📊 Total de filas: {len(df_all)}")

# Crear texto limpio
def clean_text(row):
    if isinstance(row.get("headline"), str) and row["headline"].strip() != "":
        return row["headline"].strip()
    if "word1" in row and "word2" in row:
        return f"{str(row['word1']).strip()} {str(row['word2']).strip()}"
    return ""

df_all["text_clean"] = df_all.apply(clean_text, axis=1)
df_all["text_clean"] = df_all["text_clean"].fillna("")

# ================================
# CONFIGURACIÓN DE MODELO
# ================================
candidate_labels = [
    "política", "celebridades", "tecnología", "animales",
    "comida", "deportes", "sexo", "crimen",
    "religión", "salud", "trabajo", "dinero",
    "educación", "familia", "medio ambiente",
    "ciencia", "música", "cine", "internet", "militar"
]

cols = ["lang", "headline", "word1", "word2", "text_clean", "topic_bert", "topic_score"]
output_filename = "clasificacion_BERT_parcial.csv"

st.warning("⚙️ Preparando modelo... puede tardar ⏳")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

# ================================
# CLASIFICACIÓN CON PROGRESO
# ================================
texts = df_all["text_clean"].tolist()
batch_size = 8  # seguro para CPU en Streamlit Cloud

topics = []
scores = []

progress_bar = st.progress(0)
status_text = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()
download_placeholder = st.empty()

def draw_charts(df):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    try:
        df["topic_bert"].value_counts().plot.barh(ax=axs[0], title="Distribución por tema")
        df.groupby("topic_bert")["topic_score"].mean().sort_values().plot.barh(
            ax=axs[1], title="Score promedio por tema"
        )
    except:
        pass
    plt.tight_layout()
    return fig

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

    # Actualiza DataFrame
    df_all.loc[:len(topics)-1, "topic_bert"] = topics
    df_all.loc[:len(scores)-1, "topic_score"] = scores

    df_all[cols].to_csv(output_filename, index=False, encoding="utf-8-sig")

    progress_bar.progress(len(topics) / len(texts))
    status_text.write(f"Procesadas: {len(topics)}/{len(texts)}")

    chart_placeholder.pyplot(draw_charts(df_all))
    table_placeholder.dataframe(df_all.head(10))

    download_placeholder.download_button(
        "⬇ Descargar progreso actual",
        data=open(output_filename, "rb"),
        file_name="clasificacion_BERT_parcial.csv",
        mime="text/csv"
    )

st.balloons()
st.success("🚀 Clasificación finalizada")

chart_placeholder.pyplot(draw_charts(df_all))
table_placeholder.dataframe(df_all.head(20))

st.download_button(
    "⬇ Descargar resultados finales (CSV)",
    data=open(output_filename, "rb"),
    file_name="clasificacion_BERT_final.csv",
    mime="text/csv"
)
