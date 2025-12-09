import streamlit as st
import pandas as pd
from langdetect import detect
from transformers import pipeline

# ======================
# HACKER CONSOLE UI
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
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER / CREDITS
# ======================
st.markdown("## [ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console")
st.write("Zero-shot Topic Detection + Spanish Humor Engine 🧠⚡")

st.markdown("""
📌 Designed by **Adolfo Camacho**  
🔗 [LinkedIn](https://www.linkedin.com/in/adolfo-camacho-328a2a157)  
📧 turboplay333@gmail.com  
---
""")

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_models():
    clas = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
    humor = pipeline("text-generation", model="mrm8488/t5-base-finetuned-spanish-jokes")
    return clas, humor

classifier, joke_model = load_models()
st.success("🤖 Modelos de IA cargados correctamente")

TOPICS = [
    "política", "deportes", "tecnología", "salud",
    "negocios", "cine", "ciencia", "noticias", "animales", "famosos"
]

# ======================
# FILE UPLOAD
# ======================
uploaded_file = st.file_uploader("📂 Subir archivo SemEval (CSV / TSV)", type=["csv", "tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t" if uploaded_file.name.endswith("tsv") else ",")
    st.write("🧪 Vista previa")
    st.dataframe(df.head())

    total = len(df)
    st.write(f"📦 Total de textos detectados: **{total}**")

    if st.button("🚀 Iniciar procesamiento por lotes"):
        BATCH_SIZE = 10
        processed = []
        progress_bar = st.progress(0)
        status = st.empty()
        continue_flag = True

        for i in range(0, total, BATCH_SIZE):

            if not continue_flag:
                break

            end = min(i + BATCH_SIZE, total)
            batch = df.iloc[i:end]

            st.warning(f"🔍 Analizando {i+1} → {end} de {total}…")

            # Clasificación Zero-shot
            batch_results = classifier(
                list(batch["headline"]),
                candidate_labels=TOPICS
            )

            for idx, row in batch.iterrows():
                text = str(row["headline"])

                try:
                    lang = detect(text)
                except:
                    lang = "unknown"

                topic = batch_results["labels"][idx-i][0]
                score = float(batch_results["scores"][idx-i][0])

                prompt = f"Cuéntame un chiste corto y muy gracioso en español sobre {topic}: {text}"
                joke = joke_model(prompt, max_length=50, do_sample=True)[0]["generated_text"].strip()

                processed.append({
                    "id": row["id"],
                    "headline": text,
                    "language": lang,
                    "topic": topic,
                    "score": score,
                    "joke": joke
                })

                progress_bar.progress(len(processed) / total)
                status.text(f"Procesados {len(processed)}/{total}")

            # Guardado parcial
            partial_df = pd.DataFrame(processed)
            st.download_button(
                f"⬇️ Descargar parcial {end}",
                partial_df.to_csv(index=False).encode("utf-8"),
                file_name=f"partial_{end}.csv",
                mime="text/csv",
                key=f"partial_{end}"
            )

            # Preguntar si continuar
            st.info("¿Continuar con el siguiente lote?")
            col1, col2 = st.columns(2)
            if col1.button(f"▶️ Sí {end}", key=f"yes_{end}"):
                continue_flag = True
            if col2.button(f"⏹️ No {end}", key=f"no_{end}"):
                continue_flag = False
                st.error("💣 Proceso detenido por el usuario")
                break

        # Final
        final_df = pd.DataFrame(processed)
        st.success("🎯 Procesamiento finalizado")
        st.dataframe(final_df)

        st.download_button(
            "📥 Descargar resultados finales",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="humor_output.csv",
            mime="text/csv"
        )
        st.balloons()
