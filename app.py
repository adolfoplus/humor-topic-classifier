import streamlit as st
import pandas as pd
from transformers import pipeline
import os

# UI Hacker
st.set_page_config(page_title="Humor Hacker Console", layout="wide")
st.markdown("<h2 style='color:#00FF9F;'>Humor Topic Classifier :: Hacker Console</h2>", unsafe_allow_html=True)
st.markdown("📌 Designed by <b>Adolfo Camacho</b><br>🔗 LinkedIn: adolfo-camacho-328a2a157<br>📧 turboplay333@gmail.com", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        use_auth_token=hf_token
    )

    # 👇 Nuevo modelo causal de humor español
    humor_model = pipeline(
        "text-generation",
        model="GaloSantos/fun-ES",
        use_auth_token=hf_token
    )
    return classifier, humor_model

classifier, humor_model = load_models()
st.success("🤖 Modelos cargados correctamente.")

TOPICS = ["política","deportes","tecnología","salud","negocios","cine","ciencia","noticias","animales","famosos"]

uploaded_file = st.file_uploader("📂 Subir archivo CSV/TSV", type=["csv","tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t" if uploaded_file.name.endswith(".tsv") else ",")
    st.write("Vista previa:")
    st.dataframe(df.head())

    text_col = "headline" if "headline" in df.columns else df.columns[-1]
    total = len(df)

    if st.button("🚀 Procesar"):
        results = []
        progress = st.progress(0)
        info = st.empty()

        for idx, row in df.iterrows():
            text = row[text_col]
            z = classifier(text, TOPICS)
            topic = z["labels"][0]
            score = z["scores"][0]

            # 🎭 Humor real
            prompt = f"Cuenta un chiste corto y gracioso sobre {topic}: "
            out = humor_model(prompt, max_new_tokens=40)[0]["generated_text"]
            joke = out.replace(prompt, "").strip()

            results.append({"id": idx, "text": text, "topic": topic, "score": score, "joke": joke})
            progress.progress((idx+1)/total)
            info.text(f"Procesados {idx+1}/{total}")

        out_df = pd.DataFrame(results)
        st.write("Resultado final:")
        st.dataframe(out_df)

        st.download_button("📥 Descargar CSV",
            out_df.to_csv(index=False).encode("utf-8"),
            file_name="humor_output.csv",
            mime="text/csv")

else:
    st.info("Sube un archivo para comenzar 🚀")
