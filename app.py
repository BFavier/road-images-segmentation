import streamlit as st
import pygmalion as ml
from skimage.transform import resize, rotate
import numpy as np
from skimage.io import imread

classes = ['road', 'ground', 'sky', 'building', 'vegetation', 'vehicle', 'human', 'obstacle', 'trafic sign', 'other']
colors = np.array([[200, 200, 200], [150, 80, 0], [130, 180, 255], [100, 100, 100], [0, 150, 0], [250, 100, 0], [255, 0, 0], [180, 0, 180], [255, 255, 0], [0, 0, 0]], dtype=np.uint8)


@st.cache_resource(show_spinner=False)
def download_model():
    return ml.load_model("https://drive.google.com/file/d/1s2GXAY34ScvZyFJL099GlwUcTM3dhyz8/view?usp=share_link")


@st.cache_resource(show_spinner=False)
def get_picture(file):
    return imread(file)


@st.cache_resource(show_spinner=False)
def get_default_image():
    return imread("road_image.png")


@st.cache_resource(show_spinner=False)
def resized(image):
    """
    Return the image resized and croped to a power of two dimension
    """
    if len(image.shape) == 3:
        image = image[:, :, :3]
    else:
        image = np.repeat(image[..., None], 3, -1)
    h, w = image.shape[:2]
    if h > w:
        image = rotate(image, 90.0)
        h, w = image.shape[:2]
    scale = min(h, w//2)
    H, W = scale, 2*scale
    i, j = (h - H)//2, (w - W)//2
    image = image[i:i+H, j:j+W, ...]
    image = resize(image, (256, 512), anti_aliasing=True)
    if np.all(0 <= image) and np.all(image <= 1):
        image = image * 255
    image = image.astype("uint8")
    return image


with st.sidebar:
    st.image(get_picture("picture.png"))
    st.subheader("Projet par Benoit Favier")
    st.markdown("Passionné de deep learning, j'ai implémenté des modèles pour les principales tâches de machine learning dans ma librairie [pygmalion](https://github.com/BFavier/Pygmalion) sous une licence permissive.")
    st.markdown("[Mon site web](https://bfavier.github.io/)")
    st.markdown("[Ma page GitHub](https://github.com/BFavier)")
    st.markdown("[Ma page Linkedin](https://www.linkedin.com/in/benoit-favier-9694b9206/)")

st.title("Road images segmentation")
st.markdown("This is a semantic segmentation model trained on dashcams in urban environments. It segments the image into ten classes. The image must be in landscape format.")
source = st.radio("Get image from", ('default', 'camera', 'upload'), horizontal=True)
if source == "camera":
    image_input = st.camera_input("Take a picture", label_visibility="collapsed")
    image = resized(get_picture(image_input) if image_input is not None else get_default_image())
elif source == "upload":
    file = st.file_uploader("Choose a file", label_visibility="collapsed")
    if file is not None:
        image = resized(get_picture(file))
    else:
        image = resized(get_default_image())
else:
    image = resized(get_default_image())


with st.spinner(text="Téléchargement du modèle ..."):
    model = download_model()

overlay = st.checkbox('Display overlay', value=True)

@st.cache_resource(show_spinner=False)
def colored(image: np.ndarray) -> np.ndarray:
    prediction = model.predict(image[None, ...])[0]
    dy, dx = np.diff(prediction, n=1, axis=0), np.diff(prediction, n=1, axis=1)
    boundaries = (np.pad(dy, ((1, 0), (0, 0))) | np.pad(dy, ((1, 0), (0, 0))) | np.pad(dx, ((0, 0), (1, 0))) | np.pad(dx, ((0, 0), (0, 1))))
    target_colors = colors[prediction]
    opacity = np.where(boundaries, np.array([[1.0]]), np.array([[0.7]]))[..., None]
    result = image.astype(float) * (1 - opacity) + opacity * target_colors.astype(float)
    result = np.clip(np.round(result, 0), 0, 255).astype("uint8")
    return result

st.image(colored(image) if overlay else image)


class Grid:

    def __init__(self, n_columns: int):
        self.n_columns = n_columns

    def __iter__(self):
        while True:
            for col in st.columns(self.n_columns):
                yield col

string = ""
div = []
for col, cls, color in zip(Grid(5), classes, colors):
    hexa = "#"+"".join(f"{c:04x}"[2:] for c in color)
    div.append(f"<td><font color='{hexa}'>&#9632;</font> {cls}</td>")
    if len(div) == 5:
        string += "<tr>"+" ".join(div)+"</tr>"
        div = []
st.markdown("<table>"+string+"</table>", unsafe_allow_html=True)

# for col, cls, color in zip(Grid(5), classes, colors):
#     with col:
#         # st.text(cls)
#         hexa = "#"+"".join(f"{c:04x}"[2:] for c in color)
#         st.markdown(f"<div style='color:{hexa}'>&#9632;</div>{cls}", unsafe_allow_html=True)
#         # st.color_picker(cls, "#"+"".join(f"{c:04x}"[2:] for c in color), disabled=True)

