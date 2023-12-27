import numpy as np
import requests

from PIL import Image

from src.model.obj_detect_auto import get_obj_detect_auto, preprocess_image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


def test_obj_detect_auto():
    model = get_obj_detect_auto()
    assert model is not None
    assert isinstance(model, type(EfficientNetB0(weights='imagenet')))


def test_preprocess_image():
    image_obj = Image.open(requests.get('https://storage.yandexcloud.net/mfi/1242/products/main/3474.jpg', stream=True).raw)

    img = image_obj.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    for v in preprocess_image(image_obj):
        assert v in img_array
