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


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


user_target = st.text_input('Введи название объекта (на английском) для поиска в картинке')

image_url = st.text_input('Для поиска в картинке дай ссылку на картинку в поле ниже')

image_obj = None

if image_url:
    image_obj = Image.open(requests.get(image_url, stream=True).raw)

model = load_model()

obj_on_image = []

if user_target:
    obj_on_image.extend(user_target.split(", "))

if image_obj:
    pre_img = preprocess_image(image_obj)
    preds = model.predict(pre_img)

    classes = decode_predictions(preds, top=3)[0]
    # st.write(classes)

    for cl in classes:
        if cl[2] < 0.1:
            continue

        obj_on_image.append(cl[1])
        st.write("Автоопределение:", cl[1], cl[2])

if image_obj and obj_on_image:
    st.write("Объекты которые будем искать и отмечать на картинке:", ", ".join(obj_on_image))
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

if image_obj:
    st.image(image_obj)

# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2_q3Ph31cc_MgsovOHJOKqIyTxaWnWmckLw&usqp=CAU
# https://storage.yandexcloud.net/mfi/1242/products/main/3474.jpg
# https://parkingcars.ru/wp-content/uploads/2021/02/stoyanka-1024x683.jpg
