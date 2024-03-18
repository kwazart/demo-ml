from pydantic import BaseModel


class Item(BaseModel):
    question: str
    context: str


class HistoryData:
    data: list
