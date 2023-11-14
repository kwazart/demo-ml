import requests
import torch
from PIL import Image
from transformers import pipeline

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

obj_detect_pip = pipeline(
    task="zero-shot-object-detection",
    model="google/owlvit-base-patch32",
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

result = obj_detect_pip(
    image,
    candidate_labels=["cat", "TV remote"],
)

print(result)

url = "https://www.vocord.ru/upload/iblock/bb0/bb0ecea977a089a540f2161b7157e4ef.jpg"
image = Image.open(requests.get(url, stream=True).raw)

result = obj_detect_pip(
    image,
    candidate_labels=["car"],
)

print(result)

url = "https://vestart.ru/images/2021/10/21/77777777_large.jpg"
image = Image.open(requests.get(url, stream=True).raw)

result = obj_detect_pip(
    image,
    candidate_labels=["animal"],
)

print(result)
