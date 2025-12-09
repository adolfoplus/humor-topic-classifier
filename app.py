import streamlit as st
import pandas as pd
import time
from langdetect import detect
from transformers import pipeline

# ======================
# HACKER CONSOLE UI
# ======================
st.set_page_config(page_title="Humor Hacker Console", layout="wide")

st.markdown(
    """
<style>
body, .stApp {
    background-color: black !important;
    color: #00ff99 !important;
    font-family: 'Courier New', monospace !important;
}
h1, h2, h3, h4 {
    color: #00ff99 !important;
    text-shadow: 0 0 12px #00ff99;
}
.stButton button, .stDownloadButton button {
    background-color: #002200 !important;
    color: #00ff99 !important;
    border: 1px solid #00ff99 !important;
}
.stProgress > div > div {
    background-color: #00ff99 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# HEADER
st.markdown("## [ ACCESS GRANTED ] Humor Topic Classifier ++ Joke Generator")
st.write("Zero-shot Classification + Spanish Humor Engine — Cyber Mode 🧠⚡")

st.markdown("""
📌 **Designed by Adolfo Camacho**  
🔗 [LinkedIn: adolfo-camacho-328a2a157](https://www.linkedin.com/in/adolfo-camacho-328a2a157)  
📧 turboplay333@gmail.com  
---
""")

# ======================
# MODELOS
# ======================
st.write("💾 Cargando modelos en RAM…")

classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",   # ⚡ RÁPIDO
)

joke_model = pipeline(
    "text-generation",
    model="mrm8488/t5-base-finetuned-spanish-jokes"  # Humor en español 😎
)

TOPICS = [
    "política", "deportes", "tecnología", "salud",
    "negocios", "cine", "ciencia", "noticias",
    "animales", "famosos"
]

st.success("🟢 Modelos listos para hackear la risa")

# ======================
# UTILIDADES
# ======================
def get_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def generate_spanish_joke(text, topic_es):
    prompt = f"Cuéntame un chiste corto y gracioso en español sobre {topic_es}: {text}"
    out = joke_model(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    return out.replace("\n", " ").strip()

# ======================
# ARCHIVO DE ENTRADA
# ======================
file = st.file_uploader("📂 Subir SemEval Task-A CSV o TSV", type=["csv", "tsv"])

if file:
    if file.name.endswith(".tsv"):
        df = pd.read_csv(file, sep="\t")
    else:
        df = pd.read_csv(file)

    df.columns = ["id", "word1", "word2", "headline"]
    st.dataframe(df.head())

    total = len(df)
    st.write(f"🧩 Total de textos: **{total}**")
    st.write("---")

    if st.button("🚀 Iniciar procesamiento por lotes"):
        BATCH_SIZE = 10
        progress = st.progress(0)
        status = st.empty()

        final_data = []

        continue_flag = True

        for start in range(0, total, BATCH_SIZE):
            if not continue_flag:
                break

            end = min(start + BATCH_SIZE, total)
            batch = df.iloc[start:end]

            st.warning(f"🔍 Analizando {start+1} → {end} de {total}")

            classification = classifier(list(batch["headline"]), candidate_labels=TOPICS)

            for i, row in enumerate(batch.itertuples()):
                text = row.headline
                lang = get_language(text)

                topic = classification["labels"][i][0]  # Más probable
                score = float(classification["scores"][i][0])

                joke = generate_spanish_joke(text, topic)

                final_data.append({
                    "id": row.id,
                    "headline": text,
                    "language": lang,
                    "topic": topic,
                    "score": score,
                    "joke": joke
                })

                progress.progress((len(final_data)) / total)
                status.text(f"Procesados {len(final_data)}/{total}")

            # Guardado parcial
            partial_df = pd.DataFrame(final_data)
            st.download_button(
                f"⬇️ Descargar parcial {end}",
                partial_df.to_csv(index=False).encode("utf-8"),
                f"partial_{end}.csv",
                mime="text/csv",
                key=f"p{end}"
            )

            # Preguntar si continuar
            st.info("¿Continuar con el siguiente lote?")
            col1, col2 = st.columns(2)
            if col1.button(f"▶️ Sí {end}", key=f"yes{end}"):
                continue_flag = True
            if col2.button(f"⏹️ No {end}", key=f"no{end}"):
                continue_flag = False
                st.error("⛔ Proceso detenido por el usuario.")
                break

        # Export final
        df_final = pd.DataFrame(final_data)
        st.success("🎉 Misión completada. Humor archivado.")
        st.dataframe(df_final)

        st.download_button(
            "📥 Descargar archivo final",
            df_final.to_csv(index=False).encode("utf-8"),
            "humor_output.csv",
            mime="text/csv",
        )
        st.balloons()
