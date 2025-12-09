import streamlit as st
import pandas as pd
import time
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =========================
#        HACKER UI
# =========================
st.set_page_config(page_title="🕶️ Hacker Humor Classifier", layout="wide")

st.markdown("""
<style>
body {
    background-color: black;
    color: #00ff9f;
    font-family: 'Courier New', monospace;
}
h1, h2, h3, h4 {
    color: #00ff9f;
    text-shadow: 0 0 10px #00ff9f;
}
.upload-box {
    border: 2px dashed #00ff9f;
    padding: 15px;
    text-align: center;
    background-color: #001a0f;
}
.footer {
    text-align: center;
    color: #00ffaa;
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)


# =========================
#  Load Embedding model
# =========================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embedder = load_embedder()


# =========================
#   APP UI - HEADER
# =========================
st.markdown("<h1>[ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console</h1>", unsafe_allow_html=True)
st.write("Zero-shot classifier :: Lightweight :: Batch Processing")

# =========================
#    FILE UPLOAD SECTION
# =========================
st.subheader("📥 Load CSV / TSV file")
uploaded = st.file_uploader("⬇️ Upload your SemEval Task-A File", type=["csv", "tsv"])

if uploaded:

    df = pd.read_csv(uploaded, sep="\t" if uploaded.name.endswith(".tsv") else ",")
    st.dataframe(df.head())

    if st.button("🚀 Start Batch Classification"):

        st.write("Detecting language and processing in batches of 30…")

        topics = ["política", "famosos", "noticias"]
        batch_size = 30
        results = []

        progress = st.progress(0)
        status = st.empty()

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            texts = batch["headline"].tolist()

            # embeddings
            embeddings = embedder.encode(texts)
            topic_vecs = embedder.encode(topics)

            sims = cosine_similarity(embeddings, topic_vecs)

            for j, text in enumerate(texts):
                idx = i + j
                topic_index = sims[j].argmax()
                results.append({
                    "id": batch["id"].iloc[j],
                    "text": text,
                    "topic": topics[topic_index],
                    "score": float(sims[j][topic_index])
                })

            progress.progress((i + batch_size) / len(df))
            status.write(f"✔ Batch processed: {i+batch_size}/{len(df)}")

            time.sleep(1)
            if i + batch_size < len(df):
                if st.checkbox(f"Continue after batch {i//batch_size + 1}?"):
                    pass
                else:
                    break

        st.success("🎯 Classification complete!")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.head())

        st.download_button(
            "📁 Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name="classified_results.csv"
        )


# =========================
#        FOOTER
# =========================
st.markdown("""
<div class="footer">
👨‍💻 Designed by <b>Adolfo Camacho</b><br>
🔗 <a href="https://www.linkedin.com/in/adolfo-camacho-328a2a157" style="color:#00ffaa">LinkedIn</a> |
📧 turboplay333@gmail.com
</div>
""", unsafe_allow_html=True)
