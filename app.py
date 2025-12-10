import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm

st.set_page_config(page_title="Clasificador de Humor", layout="wide")
st.title("😄 Clasificador de Temas para Task-A (Zero-Shot + Batches)")

# =============================
# Subida de archivos
# =============================
uploaded_files = st.file_uploader(
    "📂 Sube tus archivos Task-A (.tsv)",
    type=["tsv"],
    accept_multiple_files=True
)

if uploaded_files:
    dfs = {}
    for file in uploaded_files:
        lang = file.name.split("-")[-1].split(".")[0]
        df = pd.read_csv(file, sep="\t")
        df["lang"] = lang
        dfs[lang] = df

    df_all = pd.concat(dfs.values(), ignore_index=True)
    st.write("📊 Total de filas:", len(df_all))
    st.dataframe(df_all.head())

    # =============================
    # Texto limpio
    # =============================
    def clean_text(row):
        if isinstance(row.get("headline"), str) and row["headline"].strip() != "":
            return row["headline"].strip()
        if "word1" in row and "word2" in row:
            return f"{str(row['word1']).strip()} {str(row['word2']).strip()}"
        return ""

    df_all["text_clean"] = df_all.apply(clean_text, axis=1)

    if st.button("🔥 Clasificar temas"):
        st.write("⚙️ Preparando modelo...")

        # Modelo Zero-Shot Multilingüe
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli",
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
        topics = []
        scores = []

        batch_size = 8  # Ajustado para Streamlit Cloud (CPU)
        progress_bar = st.progress(0)
        status = st.empty()

        output_filename = "clasificacion_BERT_parcial.csv"
        cols = ["lang", "headline", "word1", "word2",
                "text_clean", "topic_bert", "topic_score"]

        st.write(f"🚀 Procesando {len(texts)} ejemplos por batches...\n")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            try:
                results = classifier(
                    batch_texts,
                    candidate_labels,
                    hypothesis_template="Este texto es sobre {}."
                )

                for r in results:
                    topics.append(r["labels"][0])
                    scores.append(float(r["scores"][0]))

                df_all.loc[:len(topics)-1, "topic_bert"] = topics
                df_all.loc[:len(scores)-1, "topic_score"] = scores

                df_all[cols].to_csv(output_filename,
                                    index=False,
                                    encoding="utf-8-sig")

                status.text(f"📁 Guardado parcial: {len(topics)} filas")
                progress_bar.progress(min(1, len(topics)/len(texts)))

            except Exception as e:
                st.error(f"❌ Error en batch {i}: {e}")
                break

        st.success("🎉 ¡Clasificación finalizada!")

        with open(output_filename, "rb") as f:
            st.download_button(
                label="📥 Descargar resultados (CSV)",
                data=f,
                file_name=output_filename,
                mime="text/csv"
            )
