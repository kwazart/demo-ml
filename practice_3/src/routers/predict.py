from fastapi import APIRouter
from practice_3.src.transformers.image_to_text import load_model
from practice_3.src.models.ImageIRL import ImageIRL


router = APIRouter()
image_to_text = load_model()


@router.post("/predict/")
async def predict(image_url: ImageIRL):
    return await image_to_text(image_url.url)
