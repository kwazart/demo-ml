from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


class Item(BaseModel):
    question: str
    context: str


class HistoryData:
    data: list


app = FastAPI()
model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
history = HistoryData()
history.data = list()


@app.post("/predict")
def predict(item: Item):
    input_data = {
        'question': item.question,
        'context': item.context
    }
    history.data.append(nlp(input_data))
    return nlp(input_data)


@app.get("/history")
def history():
    return history.data
