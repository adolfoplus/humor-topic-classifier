import streamlit as st
import pandas as pd
import torch
from transformers import pipeline

st.set_page_config(page_title="Clasificador de Temas Task-A", layout="wide")

st.title("😄 Clasificador de Temas para Task-A (Zero-Shot + Batches)")
st.write("📂 Sube tus archivos Task-A (.tsv)")

uploaded_files = st.file_uploader(
    "task-a-en.tsv / task-a-es.tsv / task-a-zh.tsv",
    type=["tsv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

# ---------------- Cargar corpus completo ----------------
dfs = {}
for file in uploaded_files:
    lang = file.name.split("-")[-1].split(".")[0]
    df = pd.read_csv(file, sep="\t")
    df["lang"] = lang
    dfs[lang] = df

df_all = pd.concat(dfs.values(), ignore_index=True)
st.success(f"📊 Total de filas: {len(df_all)}")

# ---------------- Texto limpio ----------------
def clean_text(row):
    if isinstance(row.get("headline"), str) and row["headline"].strip() != "":
        return row["headline"].strip()
    if "word1" in row and "word2" in row:
        return f"{str(row['word1']).strip()} {str(row['word2']).strip()}"
    return ""

df_all["text_clean"] = df_all.apply(clean_text, axis=1)
df_all["text_clean"] = df_all["text_clean"].fillna("")

candidate_labels = [
    "politics", "celebrities", "technology", "animals",
    "food", "sports", "sex", "crime",
    "religion", "health", "work", "money",
    "education", "family", "environment",
    "science", "music", "movies", "internet", "military"
]

if st.button("🔥 Clasificar temas"):
    st.warning("⚙️ Preparando modelo… puede tardar ⏳")

    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

    batch_size = 32
    topics, scores = [], []

    output_name = "clasificacion_BERT_completo.csv"
    total = len(df_all)

    progress_bar = st.progress(0)
    status = st.empty()

    for i in range(0, total, batch_size):
        batch_texts = df_all["text_clean"].iloc[i:i+batch_size].tolist()

        try:
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

            df_all.to_csv(output_name, index=False, encoding="utf-8-sig")

            progress_bar.progress(min((i+batch_size)/total, 1.0))
            status.write(f"Procesadas: {len(topics)}/{total}")

        except Exception as e:
            st.error(f"❌ Error en batch {i}: {e}")
            break

    st.success("🚀 Clasificación completada")

    st.write("📥 Descargar resultados:")
    st.download_button(
        "⬇️ Descargar CSV",
        data=open(output_name, "rb").read(),
        file_name=output_name
    )

    st.write("📌 Ejemplos clasificados:")
    st.dataframe(df_all[["text_clean", "topic_bert", "topic_score"]].head(20))
