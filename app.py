import streamlit as st
import pandas as pd
from transformers import pipeline
import time

# ======================
# CONFIG & STYLES
# ======================
st.set_page_config(page_title="Humor Hacker Console", layout="wide")

HACKER_STYLE = """
<style>
body {
    background-color: black;
    color: #00ff99;
    font-family: "Courier New", monospace;
}
h1, h2, h3, h4, h5, h6 {
    color: #00ff99;
    text-shadow: 0 0 10px #00ff99;
}
a { color: #00ffee !important; }
div.stButton > button {
    background-color: #002200;
    color: #00ff99;
    border: 1px solid #00ff99;
}
.stDownloadButton > button {
    background-color: #001a00;
    color: #00ff99 !important;
    border: 1px solid #00ff99;
}
.stProgress > div > div {
    background-color: #00ff99 !important;
}
</style>
"""
st.markdown(HACKER_STYLE, unsafe_allow_html=True)

# HEADER
st.markdown("<h1>[ ACCESS GRANTED ] Humor Topic Classifier :: Hacker Console</h1>", unsafe_allow_html=True)
st.write("Zero-shot Topic Detection + Spanish Humor Generator 💚")

st.markdown("""
📌 **Designed by Adolfo Camacho**  
🔗 [LinkedIn: adolfo-camacho-328a2a157](https://www.linkedin.com/in/adolfo-camacho-328a2a157)  
📬 turboplay333@gmail.com
""")
st.write("---")

# ======================
# LOAD MODELS
# ======================
st.write("💾 Loading AI Agents into memory… Please wait")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
lang_detector = pipeline("text-classification",
                         model="papluca/xlm-roberta-base-language-detection")
joke_model = pipeline("text-generation", model="gpt2")

st.success("🟢 Models ready for cyber-infiltration")


TOPIC_LABELS_EN = [
    "politics", "celebrities", "sports", "animals", "technology",
    "health", "business", "movies", "science", "news"
]

# ======================
# LANGUAGE & TOPIC UTILS
# ======================
def detect_lang(text):
    try:
        res = lang_detector(text[:200])
        return res[0]["label"].lower()
    except:
        return "unknown"

def translate_topic(topic):
    mapping = {
        "politics": "política", "celebrities": "famosos", "sports": "deportes",
        "animals": "animales", "technology": "tecnología", "health": "salud",
        "news": "noticias", "business": "negocios", "movies": "cine", "science": "ciencia"
    }
    return mapping.get(topic, topic)

def generate_spanish_joke(text, topic_es):
    prompt = f"Cuéntame un chiste corto y gracioso en español sobre {topic_es}: {text} ->"
    result = joke_model(prompt, max_length=60, do_sample=True, top_k=50, top_p=0.92)[0]["generated_text"]
    joke = result.replace("\n", " ").strip()
    return joke


# ======================
# FILE INPUT
# ======================
uploaded = st.file_uploader("📂 Upload SemEval Task-A CSV / TSV", type=["csv", "tsv"])

if uploaded:
    if uploaded.name.endswith(".tsv"):
        df = pd.read_csv(uploaded, sep="\t")
    else:
        df = pd.read_csv(uploaded)

    df.columns = ["id", "word1", "word2", "headline"]

    st.subheader("🧪 Preview")
    st.dataframe(df.head())


    total = len(df)
    st.write(f"🧩 Total records detected: **{total}**")

    st.write("---")

    if st.button("🚀 Launch Humor Processing (Batch Mode)"):
        BATCH_SIZE = 10
        progress = st.progress(0)

        topics_en, topics_es, scores, jokes = [], [], [], []
        continue_flag = True

        for start in range(0, total, BATCH_SIZE):
            if not continue_flag:
                break

            end = min(start + BATCH_SIZE, total)
            batch = df.iloc[start:end]

            st.warning(f"🔍 Analyzing rows {start+1} to {end} / {total} ...")

            result_batch = classifier(list(batch["headline"]), TOPIC_LABELS_EN)

            for i, text in enumerate(batch["headline"]):
                top_topic = result_batch["labels"][i][0]
                score = float(result_batch["scores"][i][0])

                topic_es = translate_topic(top_topic)
                joke = generate_spanish_joke(text, topic_es)

                topics_en.append(top_topic)
                topics_es.append(topic_es)
                scores.append(score)
                jokes.append(joke)

            # Save partial progress
            df_partial = df.iloc[:end].copy()
            df_partial["topic_en"] = topics_en
            df_partial["topic_es"] = topics_es
            df_partial["score"] = scores
            df_partial["joke"] = jokes

            progress.progress(end/total)

            st.success(f"💾 Partial export ready up to row {end}")
            st.download_button(f"⬇️ Download partial {end}",
                               df_partial.to_csv(index=False).encode("utf-8"),
                               file_name=f"partial_{end}.csv",
                               mime="text/csv")

            # Pause for user decision
            st.info("⚠️ Continue with next cyber-batch?")
            colA, colB = st.columns(2)
            if colA.button(f"▶️ Continue {end}", key=f"c{end}"):
                continue_flag = True
            if colB.button(f"⏹️ Stop {end}", key=f"s{end}"):
                st.error("🚫 Processing stopped by the operator.")
                continue_flag = False

        # === FINAL EXPORT ===
        df_final = df.copy()
        df_final["topic_en"] = topics_en
        df_final["topic_es"] = topics_es
        df_final["score"] = scores
        df_final["joke"] = jokes

        st.success("🎉 Mission Completed. Humor decrypted and archived.")
        st.download_button("⬇️ Download final humor file",
                           df_final.to_csv(index=False).encode("utf-8"),
                           file_name="humor_output.csv",
                           mime="text/csv")
        st.balloons()
