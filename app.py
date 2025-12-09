import streamlit as st
import pandas as pd
from langdetect import detect
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
st.write("Zero-shot + Spanish Humor Generator 🧠⚡")

st.markdown("""
📌 Designed by **Adolfo Camacho**  
🔗 <a href='https://www.linkedin.com/in/adolfo-camacho-328a2a157' target='_blank'>LinkedIn</a>  
📧 turboplay333@gmail.com  
---
""", unsafe_allow_html=True)

# ======================
# LOAD MODELS (CACHED)
# ======================
@st.cache_resource
def load_models():
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1"
    )
    humor_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )
    return classifier, humor_model

classifier, humor_model = load_models()
st.success("🤖 Modelos cargados correctamente")

TOPICS = [
    "política", "deportes", "tecnología", "salud",
    "negocios", "cine", "ciencia", "noticias",
    "animales", "famosos"
]

# ======================
# FILE UPLOAD
# ======================
uploaded_file = st.file_uploader("📂 Subir archivo (CSV / TSV)", type=["csv", "tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t" if uploaded_file.name.endswith("tsv") else ",")
    st.dataframe(df.head())

    total = len(df)
    st.write(f"📦 Total de registros: **{total}**")
    st.write("---")

    if st.button("🚀 Procesar por lotes"):
        BATCH_SIZE = 10
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        continue_flag = True

        for i in range(0, total, BATCH_SIZE):
            if not continue_flag:
                break

            end = min(i + BATCH_SIZE, total)
            batch = df.iloc[i:end]

            st.warning(f"🔍 Analizando {i+1} → {end} de {total}")

            # Clasificación Zero-shot
            zsc_results = classifier(
                list(batch["headline"]),
                candidate_labels=TOPICS
            )

            for idx, row in batch.iterrows():
                text = str(row["headline"])
                
                # Detección de idioma
                try:
                    lang = detect(text)
                except:
                    lang = "unknown"

                # Tema más probable
                topic = zsc_results["labels"][idx-i][0]
                score = float(zsc_results["scores"][idx-i][0])

                # Chiste corto en español
                prompt = (
                    f"Genera un chiste en español corto y muy gracioso "
                    f"sobre el tema '{topic}', con humor ingenioso."
                )
                joke_out = humor_model(prompt, max_length=60)
                joke = joke_out[0]["generated_text"].strip()

                results.append({
                    "id": row["id"],
                    "headline": text,
                    "language": lang,
                    "topic": topic,
                    "score": score,
                    "joke": joke
                })

                progress_bar.progress(len(results) / total)
                status.text(f"Procesados {len(results)}/{total}")

            # Guardado parcial
            partial_df = pd.DataFrame(results)
            st.download_button(
                f"⬇️ Descargar parcial {end}",
                partial_df.to_csv(index=False).encode("utf-8"),
                file_name=f"partial_{end}.csv",
                mime="text/csv",
                key=f"partial_{end}"
            )

            st.info("¿Continuar con el siguiente lote?")
            col1, col2 = st.columns(2)
            if col1.button(f"▶️ Sí ({end}/{total})", key=f"yes_{end}"):
                continue_flag = True
            if col2.button(f"⏹️ No ({end}/{total})", key=f"no_{end}"):
                continue_flag = False
                st.error("⛔ Proceso detenido por el usuario")
                break

        # Resultado final
        final_df = pd.DataFrame(results)
        st.success("🎯 Procesamiento finalizado")

        st.dataframe(final_df)

        st.download_button(
            "📥 Descargar resultados finales",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="humor_output.csv",
            mime="text/csv"
        )

        st.balloons()
