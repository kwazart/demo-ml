from pydantic import BaseModel


class Item(BaseModel):
    """Описание объекта, содержащего вопрос и его контекст."""
    question: str
    context: str


class HistoryData:
    """Объект для хранения истории ответов на запросы."""
    data: list
