import httpx
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO


def load_model():
    """Функция загрузки модели.

    Returns:
        image_to_text(url) -> str: асинхронная функция обработки изображения в текст.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )

    async def image_to_text(url):
        """Асинхронная функция обработки изображения в текст.
        Запрашивает изображение по URL через HTTP GET запрос.

        Args:
            url (str): URL изображения.

        Returns:
            str: текстовое описание изображения.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        raw_image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)

        return processor.decode(out[0], skip_special_tokens=True)

    return image_to_text
