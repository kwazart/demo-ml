import requests
import torch
from PIL import Image
from PIL import ImageDraw
from transformers import pipeline
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

obj_detect_pip = pipeline(
    task="zero-shot-object-detection",
    model="google/owlvit-base-patch32",
)


# @st.cache(allow_output_mutation=True)
def load_model():
    model = EfficientNetB0(weights='imagenet')
    return model


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# result = obj_detect_pip(
#     image,
#     candidate_labels=["cat", "TV remote"],
# )
#
# print(result)
#
# url = "https://www.vocord.ru/upload/iblock/bb0/bb0ecea977a089a540f2161b7157e4ef.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# result = obj_detect_pip(
#     image,
#     candidate_labels=["car"],
# )
#
# print(result)
#
# url = "https://vestart.ru/images/2021/10/21/77777777_large.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# result = obj_detect_pip(
#     image,
#     candidate_labels=["animal"],
# )
#
# print(result)

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


image_url = st.text_input('Для поиска в картинке дай ссылку на картинку в поле ниже')

image_obj = None

if image_url:
    image_obj = Image.open(requests.get(image_url, stream=True).raw)

    st.image(image_obj)

model = load_model()

if image_obj:

    x = preprocess_image(image_obj)
    preds = model.predict(x)

    classes = decode_predictions(preds, top=3)[0]

    obj_on_image = []
    for cl in classes:
        obj_on_image.append(cl[1])
        st.write(cl[1], cl[2])

    if obj_on_image:
        result = obj_detect_pip(
            image_obj,
            candidate_labels=obj_on_image,
        )

        draw = ImageDraw.Draw(image_obj)
        for prediction in result:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]

            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")

        st.image(image_obj)
