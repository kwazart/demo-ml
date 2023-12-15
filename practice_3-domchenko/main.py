import requests
import torch
from PIL import Image
from transformers import pipeline
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions
from pydantic import BaseModel
from fastapi import FastAPI

# определяем на чем будем запускать модель
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# структура запроса
class Req(BaseModel):
    url: str
    targets: str


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


app = FastAPI()


@app.get("/")
def root():
    return {'message': 'Hi there!'}


# загружаем модель для автоопределения
model = load_autodetection_model()


@app.post("/predict/")
async def predict(req: Req):
    image_obj = None

    # достаём изображение по ссылке
    if req.url:
        image_obj = Image.open(requests.get(req.url, stream=True).raw)

    obj_on_image = []

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

    # подкидываем пользовательские цели для определения
    if req.targets:
        obj_on_image.extend(req.targets.split(", "))

    # убираем дубли
    obj_on_image = list(set(obj_on_image))

    # ищем в изображении объекты по указанным целям
    result = obj_detect_pip(
        image_obj,
        candidate_labels=obj_on_image,
    )

    # возвращаем результат
    return result
