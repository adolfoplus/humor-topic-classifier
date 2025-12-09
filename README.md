# 😂 Humor Topic Classifier — BERT Zero-Shot

Webapp para clasificar temas de humor en textos de la competencia SemEval Task-A.  
Utiliza Zero-Shot Learning con BERT (`facebook/bart-large-mnli`) para asignar automáticamente categorías como política, celebridades, deportes, animales, etc.

---

## 🚀 Características

- Subida de CSV o TSV
- Limpieza automática del texto del dataset
- Clasificación de temas con Zero-Shot BERT
- Distribución visual de temas detectados
- Descarga de archivo enriquecido en CSV

---

## 🧠 Categorías analizadas

Política, celebridades, tecnología, animales, comida, deportes, sexo, crimen, religión, salud, trabajo, dinero, educación, familia, medio ambiente, ciencia, música, cine, internet, militar.

---

## ▶️ ¿Cómo ejecutar localmente?

```bash
pip install -r requirements.txt
streamlit run app.py
