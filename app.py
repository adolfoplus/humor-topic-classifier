import streamlit as st
import pandas as pd
from transformers import pipeline
import time

# ==============================
# CREDITS & HEADER UI
# ==============================
st.markdown(
    """
    <h2 style='color:#00ff9f;'>[ ACCESS GRANTED ] :: Humor Topic Classifier :: Hacker Console</h2>
    <p style='color:#00ffaa;'>Zero-shot BART :: Batch classification</p>
    <p style='color:#0088ff; font-size:14px;'>Designed by Adolfo Camacho<br>
    <a href='https://www.linkedin.com/in/adolfo-camacho-328a2a157' style='color:#00ffaa;'>LinkedIn</a> |
    <a href='mailto:turboplay333@gmail.com' style='color:#00ffaa;'>turboplay333@gmail.com</a></p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

labels = ["noticias", "famosos", "política"]

# ==============================
# FILE UPLOAD
# ==============================
uploaded = st.file_uploader("📂 Sube archivo CSV/TSV con la columna 'text'", type=["csv", "tsv"])

if uploaded:
    df = pd.read_csv(uploaded, sep="," if uploaded.name.endswith("csv") else "\t")

    if "text" not in df.columns:
        st.error("❌ ERROR: No existe la columna 'text' en tu archivo.")
        st.stop()

    st.write("📊 Vista previa")
    st.dataframe(df.head())

    process_btn = st.button("🚀 Procesar")

    if process_btn:
        results = []
        batch_size = 30
        total = len(df)

        progress = st.progress(0)
        status = st.empty()

        for i in range(0, total, batch_size):
            batch = df.iloc[i:i + batch_size]

            status.markdown(
                f"🧪 Procesando lote {i//batch_size + 1} / {total//batch_size + 1}"
            )

            for idx, row in batch.iterrows():
                text = str(row["text"])

                out = classifier(text, labels, multi_label=False)

                results.append({
                    "id": row.get("id", idx),
                    "text": text,
                    "topic": out["labels"][0],
                    "score": float(out["scores"][0])
                })

            progress.progress(min((i + batch_size) / total, 1.0))
            time.sleep(1)  # 🎬 Simulación animación hacker

            # Pregunta al usuario cada lote
            if st.button(f"Continuar después del lote {i//batch_size + 1}?"):
                pass

        output_df = pd.DataFrame(results)
        st.success("🎯 Clasificación completada!")

        st.dataframe(output_df.head(20))

        st.download_button(
            "⬇️ Descargar resultados",
            data=output_df.to_csv(index=False),
            file_name="clasificados.csv",
            mime="text/csv"
        )
