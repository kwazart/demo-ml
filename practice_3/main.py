from fastapi import FastAPI
from transformers import pipeline

from entity import HistoryData, Item

app = FastAPI()
model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
history = HistoryData()
history.data = list()


@app.post("/predict")
def predict(item: Item):
    """Получить ответ на вопрос исходя из контекста.

    Keyword arguments:
    item -- объект с вопросом и контекстом
    """
    input_data = {
        'question': item.question,
        'context': item.context
    }
    history.data.append(nlp(input_data))
    return nlp(input_data)


@app.get("/history")
def get_history():
    """Запрос истории ответов текущей сессии."""
    return history.data
