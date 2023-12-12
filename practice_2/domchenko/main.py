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

# определяем на чем будем запускать модель
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# инициализируем пайплайн с моделью
obj_detect_pip = pipeline(
    task="zero-shot-object-detection",
    model="google/owlvit-base-patch32",
)


# загружаем модель для автоопределения
def load_autodetection_model():
    model = EfficientNetB0(weights='imagenet')
    return model


# готовим изображение
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


user_target = st.text_input('Введи название объекта (на английском) для поиска в картинке')

image_url = st.text_input('Для поиска в картинке дай ссылку на картинку в поле ниже')

image_obj = None

# достаём изображение по ссылке
if image_url:
    image_obj = Image.open(requests.get(image_url, stream=True).raw)

# загружаем модель для автоопределения
model = load_autodetection_model()

obj_on_image = []

# подкидываем пользовательские цели для определения
if user_target:
    obj_on_image.extend(user_target.split(", "))

if image_obj:
    # готовим изображение для автоопределения объектов
    pre_img = preprocess_image(image_obj)
    # определяем объекты
    preds = model.predict(pre_img)

    # берём 3 наиболее вероятных определения
    classes = decode_predictions(preds, top=3)[0]

    for cl in classes:
        # откидываем определения с маленькой вероятностью
        if cl[2] < 0.1:
            continue

        obj_on_image.append(cl[1])
        # Выводим найденные объекты и вероятность определения
        st.write("Автоопределение:", cl[1], cl[2])

if image_obj and obj_on_image:
    st.write("Объекты которые будем искать и отмечать на картинке:", ", ".join(obj_on_image))

    # ищем в изображении объекты по указанным целям
    result = obj_detect_pip(
        image_obj,
        candidate_labels=obj_on_image,
    )

    # рисуем квадраты вокруг найденных объектов
    draw = ImageDraw.Draw(image_obj)
    for prediction in result:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")

# выводим изображение с отмеченными объектами
if image_obj:
    st.image(image_obj)
