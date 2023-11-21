from fastapi import FastAPI
from pydantic import BaseModel
from src.image_to_text import load_model


class Req(BaseModel):
    url: str


app = FastAPI()
image_to_text = load_model()


@app.get("/")
def root():
    return {'message': 'Hi there!'}


@app.post("/predict/")
async def predict(req: Req):
    return await image_to_text(req.url)
