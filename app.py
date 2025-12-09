import streamlit as st
import pandas as pd
import time
from transformers import pipeline
from langdetect import detect

# =============== HACKER UI STYLE ================= #
st.set_page_config(page_title="Humor Hacker Console", layout="wide")

st.markdown("""
<style>
body, .stApp {
    background-color: black !important;
    color: #00FF9F !important;
    font-family: "Courier New", monospace !important;
}
h1, h2, h3, h4 {
    color: #00FF9F !important;
}
.block-container {
    padding-top: 0rem;
}
</style>
""", unsafe_allow_html=True)

# Banner hacker
st.markdown("""
# [ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console  
Zero-shot BART + Spanish GPT :: Generating jokes with Mexican flavor...

### Designed by Adolfo Camacho  
🔗 [LinkedIn](https://www.linkedin.com/in/adolfo-camacho-328a2a157)  
📧 turboplay333@gmail.com  

---
""")

# === INPUT FILE === #
st.subheader("[ INPUT ] Load SemEval Task-A file (CSV / TSV)")

uploaded_file = st.file_uploader("Drop your CSV/TSV file here", type=["csv", "tsv"])

if uploaded_file:
    if uploaded_file.name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    classification_pipe = pipeline("zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    joke_pipe = pipeline("text-generation",
        model="mrm8488/t5-base-finetuned-spanish-jokes"
    )

    label_choices = ["politics", "sports", "technology", "health"]

    st.markdown("---")
    st.subheader("🚀 Procesando textos y generando humor...")

    total = len(df)
    batch_size = 10
    processed_jokes = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        for i, row in batch.iterrows():
            text = row["headline"]

            detected_lang = detect(text)

            # Clasificación Zero-shot
            result = classification_pipe(text, candidate_labels=label_choices)
            topic = result["labels"][0]

            # Prompt según idioma
            if detected_lang == "es":
                prompt = f"Cuéntame un chiste corto sobre {topic}:"
            else:
                prompt = f"Tell me a short joke about {topic}:"

            joke = joke_pipe(prompt, max_length=50)[0]["generated_text"]

            processed_jokes.append({
                "id": row["id"],
                "headline": text,
                "language": detected_lang,
                "topic": topic,
                "joke": joke
            })

            progress_bar.progress((i + 1) / total)
            status_text.text(f"Procesado {i+1}/{total}")

        # 🔥 Después de cada 10 → pausa para confirmar
        if end < total:
            st.warning(f"Se procesaron {end}/{total}. ¿Quieres continuar? 👀")
            cont = st.button("Continuar con los siguientes 10")
            if not cont:
                st.stop()

    res_df = pd.DataFrame(processed_jokes)
    st.success("🎉 ¡Procesamiento completado!")
    st.dataframe(res_df)

    st.download_button("📥 Descargar resultados", res_df.to_csv(index=False), "humor_results.csv")
