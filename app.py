import streamlit as st
import pandas as pd
import numpy as np
from langdetect import detect
from transformers import pipeline

# =========================
# HACKER UI CONFIG
# =========================
st.set_page_config(page_title="Humor Hacker Console", layout="wide")

# Inject CSS
st.markdown("""
<style>
body { background-color: black !important; color: #00FF9F !important; font-family: monospace; }
h1, h2, h3, h4 { color: #00FF9F !important; text-shadow: 0px 0px 9px #00FF9F; }
div.stButton > button { background-color: #001100; color: #00FF9F; border: 1px solid #00FF9F; }
.stProgress > div > div { background-color: #00FF9F; }
a { color: #00FF9F !important; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("## [ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console")
st.markdown("#### Zero-shot BART + Spanish GPT :: Generating jokes con sabor mexicano... 😎🌮")

# =========================
# CREDITOS
# =========================
st.markdown("""
<div style='color:#00FF9F; font-size:14px; margin-top: -10px;'>
Designed by <b>Adolfo Camacho</b><br>
🔗 <a href='https://www.linkedin.com/in/adolfo-camacho-328a2a157' target='_blank'>LinkedIn: adolfo-camacho-328a2a157</a><br>
📧 turboplay333@gmail.com
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS LAZY
# =========================
@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    jokemodel = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=60)
    return classifier, jokemodel

classifier, jokemodel = load_models()
st.success("🤖 Modelos de IA cargados correctamente")

# =========================
# FILE INPUT
# =========================
uploaded_file = st.file_uploader("Carga el archivo de SemEval (CSV/TSV)", type=["csv", "tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t" if uploaded_file.name.endswith("tsv") else ",")
    st.write("Vista previa:", df.head())
    
    topics_list = ["politics", "sports", "celebrity", "animals", "technology", "medical"]

    # Batch processing controls
    batch_size = 10
    total = len(df)
    progress = st.progress(0)
    results = []
    
    start = st.number_input("Iniciar desde la fila:", 0, total - 1, 0)
    
    process_btn = st.button("🚀 Iniciar procesamiento")
    
    if process_btn:
        for i in range(start, total, batch_size):

            batch_df = df.iloc[i:i+batch_size]
            for idx, row in batch_df.iterrows():
                text = str(row["headline"])

                try:
                    idioma = detect(text)
                except:
                    idioma = "unknown"

                # Clasificación de tema
                pred = classifier(text, topics_list)
                top_topic = pred["labels"][0]

                # Generación de chiste en español
                prompt = f"Genera un chiste muy corto en español relacionado con {top_topic}, que sea gracioso:"
                joke = jokemodel(prompt)[0]["generated_text"]

                results.append({
                    "id": row["id"],
                    "headline": text,
                    "topic": top_topic,
                    "joke_es": joke.strip()
                })

            # Guardado parcial
            partial = pd.DataFrame(results)
            partial.to_csv("progress_partial.csv", index=False)

            progress.progress(min((i + batch_size) / total, 1.0))

            st.write(f"⚠ Procesados: {i + batch_size}/{total}")
            st.write("📄 Guardado parcial: progress_partial.csv")

            # Pausa interactiva
            continuar = st.button(f"Continuar con el siguiente batch ({i + batch_size} → {i + 2*batch_size})")
            if not continuar:
                st.warning("⏸ Pausa activada - Puedes descargar el progreso parcial.")
                break

        # Final
        final_df = pd.DataFrame(results)
        final_df.to_csv("resultados_final.csv", index=False)

        st.success("✔ Procesamiento completado")
        st.download_button("📥 Descargar resultados", data=open("resultados_final.csv","rb"), file_name="resultados_final.csv")
        st.download_button("📥 Descargar progreso parcial", data=open("progress_partial.csv","rb"), file_name="progress_partial.csv")
