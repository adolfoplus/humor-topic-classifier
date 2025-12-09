import streamlit as st
import pandas as pd
import time
from transformers import pipeline
from langdetect import detect

# =======================
# HACKER CSS UI
# =======================
st.set_page_config(page_title="Humor Hacker Console", layout="wide")

HACKER_STYLE = """
<style>
body {
    background-color: black;
    color: #00ff99;
    font-family: "Courier New", monospace;
}
h1, h2, h3, h4 { color: #00ff99; text-shadow: 0 0 12px #00ff99; }
a { color: #00e6e6; }
div.stButton > button {
    background-color: #003300;
    color: #00ff99;
    border: 1px solid #00ff99;
}
.stDownloadButton > button {
    background-color: #001a1a;
    color: #00ff99 !important;
    border: 1px solid #00ff99;
}
.stProgress > div > div {
    background-color: #00ff99;
}
</style>
"""
st.markdown(HACKER_STYLE, unsafe_allow_html=True)

# =======================
# HEADER / CREDITOS
# =======================
st.markdown("<h1>[ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console</h1>", unsafe_allow_html=True)
st.write("Zero-shot Classification + Spanish Humor Generation — Cyber Style 💚")

st.markdown("""
📌 **Designed by Adolfo Camacho**  
🔗 [LinkedIn](https://www.linkedin.com/in/adolfo-camacho-328a2a157)  
📬 turboplay333@gmail.com
""")

st.write("---")

# =======================
# MODELOS
# =======================
st.write("🛰️ Cargando modelos…")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
joke_model = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
st.success("✔ Modelos listos")

TOPIC_LABELS_EN = [
    "politics", "celebrities", "sports", "animals", "technology", "health",
    "news", "business", "movies", "science"
]

BATCH_SIZE = 10

# =======================
# FUNCIONES
# =======================

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

def clean_joke(j):
    j = j.replace("\n", " ").strip()
    j = j.replace("Dame un chiste", "").replace("Tell me a joke", "")
    return j

def translate_topic(topic):
    mapping = {
        "politics": "Política", "celebrities": "Celebridades", "sports": "Deportes",
        "animals": "Animales", "technology": "Tecnología", "health": "Salud",
        "news": "Noticias", "business": "Negocios", "movies": "Cine", "science": "Ciencia"
    }
    return mapping.get(topic, topic)

def generate_spanish_joke(text, topic):
    prompt = f"Genera un chiste corto en español sobre {topic.lower()} relacionado con: {text} --> "
    output = joke_model(prompt, max_length=80, do_sample=True)[0]["generated_text"]
    return clean_joke(output)

# =======================
# SUBIDA DE ARCHIVO
# =======================

uploaded_file = st.file_uploader("📂 Sube tu archivo SemEval (CSV/TSV)", type=["csv", "tsv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        df = pd.read_csv(uploaded_file)

    df.columns = ["id", "word1", "word2", "headline"]
    st.write("🧪 Vista previa")
    st.dataframe(df.head())

    total_rows = len(df)

    st.write("---")
    if st.button("🔥 Iniciar procesamiento"):
        progress_bar = st.progress(0)

        topics_en, topics_es, scores, jokes = [], [], [], []

        continue_flag = True

        for start in range(0, total_rows, BATCH_SIZE):
            if not continue_flag:
                break

            end = min(start + BATCH_SIZE, total_rows)
            batch = df.iloc[start:end]

            st.write(f"🔍 Procesando textos {start+1} a {end}/{total_rows}…")

            # Clasificación
            results = classifier(list(batch["headline"]), TOPIC_LABELS_EN)

            for i, text in enumerate(batch["headline"]):
                lang = detect_language(text)

                top_topic = results["labels"][i][0]
                score = float(results["scores"][i][0])

                topic_es = translate_topic(top_topic)

                joke = generate_spanish_joke(text, topic_es)

                # Corrección de idioma
                if detect_language(joke) != "es":
                    joke = generate_spanish_joke(text, topic_es)

                topics_en.append(top_topic)
                topics_es.append(topic_es)
                scores.append(score)
                jokes.append(joke)

            # Parcial
            df_partial = df.iloc[:end].copy()
            df_partial["topic_en"] = topics_en
            df_partial["topic_es"] = topics_es
            df_partial["score"] = scores
            df_partial["joke"] = jokes

            progress_bar.progress(end / total_rows)

            st.success(f"✔ Guardado parcial hasta fila {end}")
            st.download_button(
                "⬇️ Descargar progreso parcial",
                df_partial.to_csv(index=False).encode("utf-8"),
                file_name=f"partial_{end}.csv",
                mime="text/csv",
                key=f"partial_{end}"
            )

            # Confirmación usuario
            st.write("¿Continuar con el siguiente lote?")
            col1, col2 = st.columns(2)
            if col1.button("▶️ Sí", key=f"yes_{end}"):
                continue_flag = True
            if col2.button("⏹️ No", key=f"no_{end}"):
                st.warning("🚫 Proceso detenido por el usuario.")
                continue_flag = False

        st.success("🎉 Proceso finalizado")
        st.balloons()

        # Export final
        df_final = df.copy()
        df_final["topic_en"] = topics_en
        df_final["topic_es"] = topics_es
        df_final["score"] = scores
        df_final["joke"] = jokes

        st.download_button(
            "⬇️ Descargar archivo final",
            df_final.to_csv(index=False).encode("utf-8"),
            file_name="humor_output.csv",
            mime="text/csv",
            key="final"
        )
