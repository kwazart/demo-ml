from pydantic import BaseModel


class ImageIRL(BaseModel):
    url: str
