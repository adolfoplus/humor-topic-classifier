import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# ======================
# CONFIG UI HACKER
# ======================
st.set_page_config(page_title="Humor Topic Classifier", layout="wide")

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
# HEADER + CRÉDITOS
# ======================
st.markdown("## [ ACCESS GRANTED ] Topic Classifier :: Hacker Console")
st.write("Zero-shot Topic Detection (en español) 🧠⚡")
st.markdown("""
📌 Designed by **Adolfo Camacho**  
🔗 <a href='https://www.linkedin.com/in/adolfo-camacho-328a2a157' target='_blank'>LinkedIn</a>  
📧 turboplay333@gmail.com  
---
""", unsafe_allow_html=True)

# ======================
# OPENAI
# ======================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🚨 Falta OPENAI_API_KEY en Secrets de Streamlit.")
    st.stop()

client = OpenAI(api_key=api_key)

TOPICS = [
    "política", "deportes", "tecnología", "salud",
    "negocios", "cine", "ciencia", "noticias",
    "animales", "famosos"
]

# ==================================================
# 🔎 Clasificación por Zero-Shot con OpenAI
# ==================================================
def classify_topic(text: str) -> tuple[str, float]:
    prompt = (
        "Eres un clasificador de temas en español.\n"
        f"Tópicos posibles: {', '.join(TOPICS)}\n"
        f"Texto: «{text}»\n\n"
        "Devuelve solo el tópico más adecuado de la lista."
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Clasifica titulares en uno de los tópicos dados."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=16
    )

    label = resp.choices[0].message.content.strip().lower()
    best = "desconocido"
    for t in TOPICS:
        if t.lower() in label:
            best = t
            break

    return best, 1.0 if best != "desconocido" else 0.0


# ======================
# CARGA DEL ARCHIVO
# ======================
uploaded_file = st.file_uploader("📂 Subir archivo SemEval (CSV / TSV)", type=["csv", "tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t" if uploaded_file.name.endswith(".tsv") else ",")
    st.subheader("🧪 Vista previa")
    st.dataframe(df.head())

    # Detectar columna de texto automáticamente
    text_col = "headline" if "headline" in df.columns else df.columns[-1]

    total = len(df)
    st.write(f"📦 Registros totales: **{total}**")
    st.write("---")

    if st.button("🚀 Clasificar por lotes de 30"):
        BATCH = 30
        results = []
        progress = st.progress(0)
        info = st.empty()

        for start in range(0, total, BATCH):
            end = min(start + BATCH, total)
            batch = df.iloc[start:end]

            st.warning(f"🔍 Procesando filas {start+1} a {end} de {total}…")

            for idx, row in batch.iterrows():
                text = str(row[text_col])
                topic, score = classify_topic(text)

                results.append({
                    "id": row.get("id", idx),
                    "text": text,
                    "topic": topic,
                    "score": score
                })

                progress.progress(len(results) / total)
                info.text(f"Procesados {len(results)}/{total}")

            # Descargar parcial por batch
            partial_df = pd.DataFrame(results)
            st.download_button(
                f"⬇️ Descargar parcial hasta {end}",
                partial_df.to_csv(index=False).encode("utf-8"),
                file_name=f"partial_{end}.csv",
                mime="text/csv",
                key=f"partial_{end}"
            )

        final_df = pd.DataFrame(results)
        st.success("🎯 Clasificación completada")
        st.dataframe(final_df)

        st.download_button(
            "📥 Descargar resultados finales",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="topics_output.csv",
            mime="text/csv"
        )

        st.balloons()
else:
    st.info("Sube un archivo CSV/TSV para comenzar 🚀")
