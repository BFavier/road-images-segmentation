import streamlit as st
import pygmalion as ml
import re
import string
from typing import List
from itertools import zip_longest
from skimage.io import imread


@st.cache_resource(show_spinner=False)
def download_model():
    return ml.load_model("https://drive.google.com/file/d/1l4jxLJpLt8xmxf9JMIVM4lYv42erF--0/view?usp=share_link")


@st.cache_resource(show_spinner=False)
def get_picture():
    return imread("picture.png")


def capitalize(string: str) -> str:
    return string[0].upper() + string[1:]


def format_sentences(text: str) -> List[str]:
    """
    Split an input text into formated input sentences
    """
    pattern = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+"
    sentences = re.split(pattern, text)
    spacers = re.findall(pattern, text)
    sentences = (capitalize(s).strip() for s in sentences)
    sentences = (s.replace(" ,", ",").replace(" .", ".") for s in sentences)
    sentences = (re.sub("(?<=[^\\s\\^]):", " :", re.sub("(?<=[^\\s\\^])\?", " ?", re.sub("(?<=[^\\s\\^])!", " !", s))) for s in sentences)
    sentences = [s if s.endswith((".", "?", "!", ":")) else s+"." for s in sentences]
    return sentences, spacers

with st.sidebar:
    st.image(get_picture())
    st.subheader("Projet par Benoit Favier")
    st.markdown("Passionné de deep learning, j'ai implémenté des modèles pour les principales tâches de machine learning dans ma librairie [pygmalion](https://github.com/BFavier/Pygmalion) sous une licence permissive.")
    st.markdown("[Mon site web](https://bfavier.github.io/)")
    st.markdown("[Ma page GitHub](https://github.com/BFavier)")
    st.markdown("[Ma page Linkedin](https://www.linkedin.com/in/benoit-favier-9694b9206/)")

st.title("Traduction français → anglais")
st.markdown("Cette application est une démonstration d'un modèle de NMT (Neural Machine Translation). "
            "La traduction est effectuée par un modèle de machine learning (Transformer) "
            "entraîné sur ~3 millions de paires de phrases français/anglais. "
            "Le modèle est appliqué phrase par phrase, sans tenir compte du contexte du document entier.")
st.markdown("Vous pouvez par exemple le tester avec des paragraphes issus d'une page aléatoire de [Wikipedia](https://fr.wikipedia.org/wiki/Sp%C3%A9cial:Page_au_hasard).")
st.subheader("Le texte à traduire:")
input_text = st.text_area(label="Le texte à traduire:",
                          placeholder="Pinocchio avait toujours voulu devenir un véritable petit garçon...",
                          max_chars=10000, height=200, label_visibility="collapsed")

with st.spinner(text="Téléchargement du modèle ..."):
    model = download_model()

if len(input_text) > 0:
    inputs, spacers = format_sentences(input_text)
    predictions = []
    progress_bar = st.progress(0., "Application du modèle ...")
    for i, input in enumerate(inputs, start=1):
        predictions.append(model.predict(input, n_beams=3)[0])
        progress_bar.progress(int(i*100/len(inputs)), "Traduction en cours ...")
    predictions = ["".join(filter(lambda x: x in string.printable, s)) for s in predictions]  # Filtre les charactères non-affichables
    progress_bar.empty()
    st.caption("Traduction:")
    translation = "".join(s for t in zip_longest(predictions, spacers, fillvalue="") for s in t)
    st.markdown(translation)
