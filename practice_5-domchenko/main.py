from fastapi import FastAPI

from src.handler.handler import handler, Req

app = FastAPI()


# метод приветствия для проверки работы сервера
@app.get("/")
def root():
    return {'message': 'Hello World!!!'}


# метод поиска заданных объектов в картинке по ссылке
@app.post("/predict/")
async def predict(req: Req):
    return handler(req)
