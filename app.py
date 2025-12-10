import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ================================
# 🔹 CONFIGURACIÓN DEL MODELO
# ================================
topics = [
    "noticias", "deportes", "famosos", "politica", "tecnologia",
    "economia", "ciencia", "salud", "entretenimiento", "medio ambiente"
]

classifier = pipeline(
    "zero-shot-classification",
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ================================
# 🔹 CLASIFICACIÓN POR LOTES
# ================================
def classify_batch(df):
    texts = df["text"].tolist()
    results = classifier(texts, candidate_labels=topics)

    df["topic"] = [res["labels"][0] for res in results]
    df["score"] = [float(res["scores"][0]) for res in results]

    return df


# ================================
# 🔹 GRÁFICA POR BATCH
# ================================
def show_pie_chart(df, batch_index):
    count_topics = df["topic"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(count_topics, labels=count_topics.index, autopct="%1.1f%%")
    ax.set_title(f"Distribución de temas - Lote {batch_index+1}")
    st.pyplot(fig)


# ================================
# 🔹 INTERFAZ STREAMLIT
# ================================
st.title("Clasificador de Temas por Bloques")

uploaded_file = st.file_uploader("📄 Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📌 Datos cargados:")
    st.dataframe(df.head())

    if st.button("🚀 Procesar en bloques de 100"):
        processed = pd.DataFrame()
        total_rows = len(df)
        num_batches = total_rows // 100 + (1 if total_rows % 100 != 0 else 0)

        for i in range(num_batches):
            st.write(f"🔹 Procesando lote {i+1}/{num_batches}...")
            batch_df = df.iloc[i*100:(i+1)*100].copy()
            batch_df = classify_batch(batch_df)
            processed = pd.concat([processed, batch_df])

            # Mostrar gráfica de cada lote
            show_pie_chart(batch_df, i)

            st.write(batch_df.head())

        st.success("🎯 Clasificación completada de todos los lotes!")

        # Botón para descargar resultados
        csv_out = processed.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar CSV Resultado",
            data=csv_out,
            file_name="clasificado.csv",
            mime="text/csv"
        )
