import httpx
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO


def load_model():
    """The function loads a model describing the passed image, and returns the function image_to_text(img_url), to pass its URL to the image."""

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large")

    async def image_to_text(url):
        """The function takes a URL to a image and returns a description of it."""

        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        raw_image = Image.open(BytesIO(response.content)).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)

        return processor.decode(out[0], skip_special_tokens=True)

    return image_to_text
