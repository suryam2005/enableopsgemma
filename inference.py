from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

base_model_id = "google/gemma-2b-it"
adapter_path = "./gemma-enableops"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True).cpu()
model = PeftModel.from_pretrained(base_model, adapter_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt: str):
    result = pipe(prompt, max_new_tokens=100, do_sample=True)
    return result[0]["generated_text"]
