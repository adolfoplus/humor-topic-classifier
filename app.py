import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# =========================
# Modelo Zero-Shot
# =========================
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli"
    )

classifier = load_classifier()

TOPICS = ["noticias", "política", "famosos"]


# =========================
# Función de clasificación
# =========================
def classify_batch(texts):
    preds = []
    for t in texts:
        try:
            result = classifier(t, TOPICS)
            preds.append(result["labels"][0])
        except:
            preds.append("error")
    return preds


# =========================
# UI Streamlit
# =========================
st.title("Clasificador de Temas (Humor Task)")
st.write("📌 Procesa cada 100 ejemplos y muestra una gráfica por lote.")


uploaded = st.file_uploader("Sube tu archivo .tsv", type=["tsv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, sep="\t")

    # 🔴 Validación correcta
    if "headline" not in df.columns:
        st.error("El archivo debe tener una columna llamada 'headline'")
        st.stop()

    texts = df["headline"].fillna("").tolist()

    st.success("Archivo correcto ✔️ ¡Listo para clasificar!")

    results = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        st.write(f"👉 Clasificando lote {i} – {i+len(batch)}...")

        batch_preds = classify_batch(batch)
        results.extend(batch_preds)

        batch_df = pd.DataFrame({"headline": batch, "topic": batch_preds})

        st.write(batch_df.sample(min(5, len(batch_df))))

        # =========================
        # Gráfica de pastel
        # =========================
        counts = batch_df["topic"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="pie", autopct='%1.1f%%', startangle=90, ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Distribución de temas en lote {i}")
        st.pyplot(fig)

    # Se muestran TODOS los resultados al final
    df["topic"] = results
    st.write("📊 Resultados completos:")
    st.dataframe(df)

    st.download_button(
        "📥 Descargar resultados",
        df.to_csv(index=False).encode("utf-8"),
        "resultados_clasificados.csv",
        "text/csv"
    )
