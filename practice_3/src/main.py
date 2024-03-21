from fastapi import FastAPI
from practice_3.src.routers import predict, root


app = FastAPI()

app.include_router(root.router)
app.include_router(predict.router)
