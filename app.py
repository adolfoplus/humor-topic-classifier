import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
import time
import os

st.set_page_config(page_title="Humor Topic Classifier", layout="wide", page_icon="😂")

# ====== CSS Premium Refinado ======
st.markdown("""
<style>
/* Title */
h1 {
    text-align:center;
    font-weight:900 !important;
    font-size:32px !important;
    color:#00ADB5 !important;
}

/* Upload box center-align */
.block-container {
    padding-top: 2rem !important;
}

/* Card style */
div[data-testid="stFileUploader"] {
    border: 2px dashed #00ADB5 !important;
    border-radius: 15px !important;
    padding: 25px !important;
}

/* Buttons */
button {
    border-radius: 12px !important;
    font-weight:600 !important;
}

/* Footer */
footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Logo + Title center
st.markdown("<h1>😂 Humor Topic Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#AAAAAA;'>Zero-Shot BERT • SemEval Humor Task</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📂 Drag or Browse your CSV/TSV file", type=["csv", "tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.markdown("### 📊 Preview of Input Data")
    st.dataframe(df.head())

    def build_text(row):
        if "headline" in df.columns and isinstance(row["headline"], str) and row["headline"].strip() != "-":
            return row["headline"].strip()
        if "word1" in df.columns and "word2" in df.columns:
            return f"{str(row['word1'])} {str(row['word2'])}".strip()
        return ""

    df["text_clean"] = df.apply(build_text, axis=1)

    st.markdown("### 🧠 Loading Zero-Shot model…")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    st.success("Model Loaded ✔")

    labels = [
        "política","celebridades","tecnología","animales","comida",
        "deportes","sexo","crimen","religión","salud",
        "trabajo","dinero","educación","familia","medio ambiente",
        "ciencia","música","cine","internet","militar"
    ]

    texts = df["text_clean"].tolist()
    total = len(texts)
    batch_size = 32
    topics, scores = [], []
    progress_bar = st.progress(0)
    status = st.empty()
    logs = st.container()
    start_time = time.time()

    out_csv = "parcial_clasificacion.csv"

    st.markdown("### 🔄 Classifying…")

    try:
        for i in range(0, total, batch_size):
            res = classifier(
                texts[i:i+batch_size],
                labels,
                hypothesis_template="Este texto es sobre {}."
            )

            for r in res:
                topics.append(r["labels"][0])
                scores.append(r["scores"][0])

            df.loc[:len(topics)-1,"topic_bert"]=topics
            df.loc[:len(scores)-1,"topic_score"]=scores
            df.to_csv(out_csv,index=False)

            elapsed=time.time()-start_time
            progress=(i+batch_size)/total
            eta=elapsed/progress-elapsed if progress>0 else 0

            progress_bar.progress(progress)
            status.info(f"✔ {i+batch_size}/{total} • {progress*100:.1f}% • "
                        f"⏱ {elapsed/60:.1f}m • ETA {eta/60:.1f}m")

            with logs:
                st.write(f"🟦 Progress: {i+batch_size} rows processed")

            st.download_button(
                "📥 Download Partial",
                df.to_csv(index=False).encode("utf-8"),
                "partial_result.csv",
                key=f"p{i}"
            )

        status.success("🎉 Classification Complete")

    except Exception as e:
        st.error("❌ Error occurred — Partial saved")

    st.markdown("### 📈 Topic Distribution")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(
        data=df[df["topic_bert"].notna()],
        y="topic_bert",
        order=df["topic_bert"].value_counts().index,
        palette="turbo"
    )
    st.pyplot(fig)

    st.download_button(
        "📥 Download Final Result CSV",
        df.to_csv(index=False).encode("utf-8"),
        "final_output.csv"
    )

# Footer
st.markdown(
    "<br><br><p style='text-align:center; color:#777;'>Designed by "
    "<strong style='color:#00ADB5;'>Adolfo Camacho</strong></p>",
    unsafe_allow_html=True
)
