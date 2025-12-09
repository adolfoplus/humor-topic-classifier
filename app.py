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
        model="mrm8488/t5-small-spanish-jokes"  # 🔥 Modelo ligero de humor
    )
    return classifier, humor_model

classifier, humor_model = load_models()
st.success("🤖 Modelos cargados correctamente.")

TOPICS = [
    "política", "deportes", "tecnología", "salud",
    "negocios", "cine", "ciencia", "noticias",
    "animales", "famosos"
]

# ======================
# FILE UPLOAD
# ======================
uploaded_file = st.file_uploader("📂 Subir archivo SemEval (CSV / TSV)", type=["csv","tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t" if uploaded_file.name.endswith(".tsv") else ",")
    st.subheader("🧪 Vista previa")
    st.dataframe(df.head())

    text_col = "headline" if "headline" in df.columns else df.columns[-1]

    total = len(df)
    st.write(f"📦 Total de registros: **{total}**")
    st.write("---")

    if st.button("🚀 Procesar por lotes (10 en 10)"):
        BATCH = 10
        results = []
        progress = st.progress(0)
        info = st.empty()

        for start in range(0, total, BATCH):
            end = min(start + BATCH, total)
            batch = df.iloc[start:end]

            st.warning(f"🔍 Analizando filas {start+1} a {end} de {total}…")

            texts = batch[text_col].astype(str).tolist()
            zsc = classifier(texts, candidate_labels=TOPICS)

            for i, row in enumerate(batch.itertuples()):
                # ZERO-SHOT seguro
                try:
                    topic = zsc[i]["labels"][0]
                    score = float(zsc[i]["scores"][0])
                except Exception:
                    topic = "desconocido"
                    score = 0.0

                # 🎭 CHISTE ESPAÑOL
                prompt = f"Cuento algo gracioso sobre {topic}:"
                joke = humor_model(prompt, max_length=50)[0]["generated_text"].strip()

                results.append({
                    "id": getattr(row, "id", row.Index),
                    "text": getattr(row, text_col),
                    "topic": topic,
                    "score": score,
                    "joke": joke
                })

                progress.progress(len(results) / total)
                info.text(f"Procesados {len(results)}/{total}")

            partial_df = pd.DataFrame(results)
            st.download_button(
                f"⬇️ Descargar parcial hasta {end}",
                partial_df.to_csv(index=False).encode("utf-8"),
                file_name=f"partial_{end}.csv",
                mime="text/csv",
                key=f"partial_{end}"
            )

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
    st.info("Sube un archivo CSV/TSV para comenzar 🚀")
