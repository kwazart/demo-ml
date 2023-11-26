from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel
  
tokenizer = AutoTokenizer.from_pretrained("nbroad/mt5-small-qgen")
model = AutoModelForSeq2SeqLM.from_pretrained("nbroad/mt5-small-qgen")


class Text(BaseModel):
    input: str
    
app = FastAPI()

@app.get("/")
def root():
    return {"Приветствую тебя!"}

@app.post("/predict/")
def predict(str: Text):
    inputs = tokenizer(str.input, return_tensors="pt")
    output = model.generate(**inputs, max_length=40)
    return tokenizer.decode(output[0], skip_special_tokens=True)
