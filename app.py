import streamlit as st
import pygmalion as ml
from skimage.transform import resize
import numpy as np
from typing import List
from skimage.io import imread

classes = ['road', 'ground', 'sky', 'building', 'vegetation', 'vehicle', 'human', 'obstacle', 'trafic sign', 'other']
colors = np.array([[200, 200, 200], [150, 80, 0], [130, 180, 255], [100, 100, 100], [0, 150, 0], [250, 100, 0], [255, 0, 0], [180, 0, 180], [255, 255, 0], [0, 0, 0]], dtype=np.uint8)


@st.cache_resource(show_spinner=False)
def download_model():
    return ml.load_model("https://drive.google.com/file/d/1s2GXAY34ScvZyFJL099GlwUcTM3dhyz8/view?usp=share_link")


@st.cache_resource(show_spinner=False)
def get_picture():
    return imread("picture.png")


@st.cache_resource(show_spinner=False)
def get_default_image():
    return imread("place.jpg")


@st.cache_resource(show_spinner=False)
def resized(image):
    """
    Return the image resized and croped to a power of two dimension
    """
    h, w = image.shape[:2]
    H, W = (2**np.floor([np.log2(h), np.log2(w)])).astype(int)
    i, j = (h - H)//2, (w - W)//2
    image = image[i:i+H, j:j+W, ...]
    image = resize(image, (256, 512), anti_aliasing=True)
    if np.all(0 <= image) and np.all(image <= 1):
        image = image * 255
    image = image.astype("uint8")
    return image


with st.sidebar:
    st.image(get_picture())
    st.subheader("Projet par Benoit Favier")
    st.markdown("Passionné de deep learning, j'ai implémenté des modèles pour les principales tâches de machine learning dans ma librairie [pygmalion](https://github.com/BFavier/Pygmalion) sous une licence permissive.")
    st.markdown("[Mon site web](https://bfavier.github.io/)")
    st.markdown("[Ma page GitHub](https://github.com/BFavier)")
    st.markdown("[Ma page Linkedin](https://www.linkedin.com/in/benoit-favier-9694b9206/)")

st.title("Road images segmentation")
st.markdown(".")
image = resized(st.camera_input("Take a picture") or get_default_image())


with st.spinner(text="Téléchargement du modèle ..."):
    model = download_model()


@st.cache_resource(show_spinner=False)
def colored(image: np.ndarray) -> np.ndarray:
    prediction = model.predict(image[None, ...])[0]
    dy, dx = np.diff(prediction, n=1, axis=0), np.diff(prediction, n=1, axis=1)
    boundaries = (np.pad(dy, ((1, 0), (0, 0))) | np.pad(dy, ((1, 0), (0, 0))) | np.pad(dx, ((0, 0), (1, 0))) | np.pad(dx, ((0, 0), (0, 1))))
    target_colors = colors[prediction]
    opacity = np.where(boundaries, np.array([[1.0]]), np.array([[0.5]]))[..., None]
    result = image.astype(float) * (1 - opacity) + opacity * target_colors.astype(float)
    result = np.clip(np.round(result, 0), 0, 255).astype("uint8")
    return result

col1, col2, col3 = st.columns(3)

with col1:
    st.write('')

with col2:
    st.image(colored(image))

with col3:
    st.write('')

