# !pip install numpy tensorflow tensorflow_hub imageio ipython absl-py streamlit
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import imageio
import streamlit as st
import base64
from absl import logging

latent_dim = 512


def interpolate_hypersphere(v1, v2, num_steps):
    v1_norm = tf.norm(v1)
    v2_norm = tf.norm(v2)
    v2_normalized = v2 * (v1_norm / v2_norm)
    vectors = []
    for step in range(num_steps):
        interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
        interpolated_norm = tf.norm(interpolated)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        vectors.append(interpolated_normalized)
    return tf.stack(vectors)


def animate(images):
    images = np.array(images)
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    path = './animation.gif'
    imageio.mimsave(path, converted_images)
    return path


logging.set_verbosity(logging.ERROR)

progan = hub.load(
    "https://tfhub.dev/google/progan-128/1").signatures['default']


def interpolate_between_vectors(seed):
    tf.random.set_seed(seed)
    v1 = tf.random.normal([latent_dim])
    v2 = tf.random.normal([latent_dim])
    vectors = interpolate_hypersphere(v1, v2, 50)
    interpolated_images = progan(vectors)['default']
    return interpolated_images


def print_gif(url):
    file_ = open(url, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="face gif">',
                unsafe_allow_html=True)


if 'seed' not in st.session_state:
    st.session_state.seed = 0

if st.button('Интерполяция'):
    st.session_state.seed += 1
    interpolated_images = interpolate_between_vectors(st.session_state.seed)
    path = animate(interpolated_images)
    print_gif(path)
