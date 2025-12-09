import streamlit as st
import pandas as pd
from transformers import pipeline

# ======================
# HACKER UI CONFIG
# ======================
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
    text-shadow: 0 0 8px #00FF9F;
}
.stButton button, .stDownloadButton button {
    background-color: #002200 !important;
    color: #00FF9F !important;
    border: 1px solid #00FF9F !important;
}
.stProgress > div > div {
    background-color: #00FF9F !important;
}
a {
    color: #00FF9F !important;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER + CREDITOS
# ======================
st.markdown("## [ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console")
st.write("Zero-shot Topic Detection + Spanish Humor Generator 🧠⚡")

st.markdown("""
📌 Designed by **Adolfo Camacho**  
🔗 <a href='https://www.linkedin.com/in/adolfo-camacho-328a2a157' target='_blank'>LinkedIn</a>  
📧 turboplay333@gmail.com  
---
""", unsafe_allow_html=True)

# ======================
# MODELOS (CACHÉ)
# ======================
@st.cache_resource
def load_models():
    # Zero-shot más ligero que bart-large
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1"
    )

    # Modelo de texto en español (no requiere sentencepiece)
    humor_model = pipeline(
        "text-generation",
        model="flax-community/spanish-gpt2-small"
    )
    return classifier, humor_model

st.info("Cargando modelos… esto puede tardar un poco la primera vez.")
classifier, humor_model = load_models()
st.success("🤖 Modelos cargados correctamente")

TOPICS = [
    "política", "deportes", "tecnología", "salud",
    "negocios", "cine", "ciencia", "noticias",
    "animales", "famosos"
]

# ======================
# FUNCIONES AUXILIARES
# ======================
def detect_topic_batch(texts):
    """
    Corre zero-shot sobre una lista de textos.
    Devuelve lista de (topic, score).
    """
    res = classifier(
        texts,
        candidate_labels=TOPICS,
        hypothesis_template="Este texto es sobre {}."
    )
    topics = []
    for r in res:
        topics.append((r["labels"][0], float(r["scores"][0])))
    return topics

def generate_spanish_joke(topic, text):
    """
    Genera un chiste corto en español usando un GPT-2 entrenado en español.
    """
    prompt = (
        f"Escribe un chiste corto y muy gracioso en español "
        f"sobre el tema '{topic}'. Que sea ingenioso y original."
    )
    out = humor_model(
        prompt,
        max_length=60,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )[0]["generated_text"]
    joke = out.replace("\n", " ").strip()
    return joke

# ======================
# FILE UPLOAD
# ======================
uploaded_file = st.file_uploader("📂 Subir archivo SemEval Task A (CSV / TSV)", type=["csv", "tsv"])

if uploaded_file:
    # Cargar TSV o CSV
    if uploaded_file.name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("🧪 Vista previa de los datos")
    st.dataframe(df.head())

    # Intentar identificar columna de texto
    text_col = None
    for candidate in ["headline", "text", "sentence"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # si no encontramos, usamos la última columna
        text_col = df.columns[-1]

    st.write(f"📌 Columna de texto usada para análisis: **{text_col}**")

    total = len(df)
    st.write(f"📦 Total de registros: **{total}**")
    st.write("---")

    if st.button("🚀 Iniciar procesamiento completo (batch de 10)"):
        BATCH_SIZE = 10
        results = []
        progress_bar = st.progress(0)
        status = st.empty()

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = df.iloc[start:end]

            st.warning(f"🔍 Analizando {start+1} → {end} de {total}…")

            texts = [str(t) for t in batch[text_col].tolist()]
            # Clasificación por batch
            topic_batch = detect_topic_batch(texts)

            for (idx, row), (topic, score) in zip(batch.iterrows(), topic_batch):
                text = str(row[text_col])

                joke = generate_spanish_joke(topic, text)

                result_row = {
                    "id": row[df.columns[0]] if "id" in df.columns else idx,
                    "text": text,
                    "topic": topic,
                    "score": score,
                    "joke": joke
                }
                results.append(result_row)

                progress_bar.progress(len(results) / total)
                status.text(f"Procesados {len(results)}/{total}")

            # Guardado parcial y botón de descarga
            partial_df = pd.DataFrame(results)
            st.download_button(
                f"⬇️ Descargar parcial hasta {end}",
                partial_df.to_csv(index=False).encode("utf-8"),
                file_name=f"partial_{end}.csv",
                mime="text/csv",
                key=f"partial_{end}"
            )

        # Final
        final_df = pd.DataFrame(results)
        st.success("🎯 Procesamiento completado")
        st.dataframe(final_df)

        st.download_button(
            "📥 Descargar resultados finales",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="humor_output.csv",
            mime="text/csv"
        )

        st.balloons()
else:
    st.info("Sube un archivo CSV o TSV para comenzar el análisis.")
