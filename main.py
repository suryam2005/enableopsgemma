from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = FastAPI()

# Load from Hugging Face
model_id = "suryamuralirajan/enableops-gemma"
token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=token)

class PromptInput(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(data: PromptInput):
    inputs = tokenizer(data.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}
