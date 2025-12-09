import streamlit as st
import pandas as pd
import time
import torch
from transformers import pipeline

st.set_page_config(
    page_title="Humor Topic Classifier 😂",
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====== ESTILOS PREMIUM ======
st.markdown("""
<style>
body {
    background-color: #111;
}
table tbody tr:hover {
    background-color: #222 !important;
}
.stProgress > div > div {
    background-color: #49a6ff;
}
h1, h2, h3, h4, h5, h6, p, span, li {
    color: #eee;
}
</style>
""", unsafe_allow_html=True)


st.title("😂 Humor Topic Classifier + Joke Generator")

st.write("Sube tu archivo CSV/TSV del Task-A:")

uploaded_file = st.file_uploader("Drag and drop file here", type=["csv", "tsv"], label_visibility="collapsed")
resume = st.checkbox("🔄 Reanudar desde progreso_parcial.csv (si existe)", value=True)

if uploaded_file:
    sep = "," if uploaded_file.name.endswith(".csv") else "\t"
    df_all = pd.read_csv(uploaded_file, sep=sep)
    if "headline" not in df_all.columns:
        st.error("❌ La columna 'headline' es obligatoria")
        st.stop()

    st.subheader("👀 Vista previa")
    st.dataframe(df_all.head(), use_container_width=True)

    # ====== CARGA DE MODELOS ======
    with st.spinner("🤖 Cargando modelos de IA..."):
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        generator = pipeline(
            "text-generation",
            model="gpt2",
            device=0 if torch.cuda.is_available() else -1,
            pad_token_id=50256
        )
    st.success("🤖 Modelos cargados ✔")

    labels = ["política", "celebridades", "deportes", "animales",
              "salud", "comida", "tecnología", "religión", "crimen"]


    st.subheader("🚀 Procesando y generando humor...")

    progress_bar = st.progress(0)
    status_text = st.empty()
    table_preview = st.empty()
    partial_download_btn = st.empty()

    topics = []
    scores = []
    jokes = []

    TOTAL = len(df_all)

    # Reanudación si aplica
    initial_index = 0
    if resume and "progress_partial.csv" in st.session_state:
        saved = pd.read_csv("progress_partial.csv")
        already = len(saved)
        topics = saved["topic"].tolist()
        scores = saved["score"].tolist()
        jokes = saved["joke"].tolist()
        initial_index = already
        st.info(f"🔄 Reanudando desde {already}/{TOTAL}")
    elif resume and uploaded_file.name != "progress_partial.csv":
        try:
            saved = pd.read_csv("progress_partial.csv")
            already = len(saved)
            topics = saved["topic"].tolist()
            scores = saved["score"].tolist()
            jokes = saved["joke"].tolist()
            initial_index = already
            st.info(f"🔄 Reanudando desde {already}/{TOTAL}")
        except:
            pass


    # ====== LOOP PRINCIPAL ======
    for i in range(initial_index, TOTAL):
        text = df_all.loc[i, "headline"]

        # --- Clasificación
        result = classifier(text, labels)
        topic = result["labels"][0]
        score = result["scores"][0]
        if hasattr(score, "detach"):
            score = score.detach().cpu().numpy()
        topics.append(topic)
        scores.append(float(score))

        # --- Generación de chiste
        prompt = f"Dime un chiste corto sobre {topic}: \"{text}\""
        joke_result = generator(prompt, max_length=60, num_return_sequences=1)
        jokes.append(joke_result[0]["generated_text"])

        # Guardado incremental
        df_partial = df_all.iloc[: i+1].copy()
        df_partial["topic"] = topics
        df_partial["score"] = scores
        df_partial["joke"] = jokes
        df_partial.to_csv("progress_partial.csv", index=False)

        # === UI EN VIVO ===
        progress = int((i+1) / TOTAL * 100)
        progress_bar.progress(progress)
        status_text.write(f"Procesado {i+1}/{TOTAL}")

        # Últimas 20 filas dinámicas
        table_preview.dataframe(df_partial.tail(20), use_container_width=True)

        # Botón de descarga parcial
        partial_download_btn.download_button(
            label=f"📥 Descargar resultados ({i+1}/{TOTAL})",
            data=df_partial.to_csv(index=False).encode("utf-8"),
            file_name="progreso_parcial.csv",
            mime="text/csv",
            key=f"partial_{i}"
        )

    # ====== FINAL ======
    st.balloons()
    st.success("🎉 ¡Completado!")

    df_final = df_all.copy()
    df_final["topic"] = topics
    df_final["score"] = scores
    df_final["joke"] = jokes
    df_final.to_csv("final_results.csv", index=False)

    st.download_button(
        label="📦 Descargar archivo final (final_results.csv)",
        data=df_final.to_csv(index=False).encode("utf-8"),
        file_name="final_results.csv",
        mime="text/csv"
    )

    st.subheader("📊 Distribución de temas")
    st.bar_chart(df_final["topic"].value_counts())
