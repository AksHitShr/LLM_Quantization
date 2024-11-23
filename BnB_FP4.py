import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import math
import time
import bitsandbytes as bnb

def calculate_perplexity(model, tokenizer, text, max_length=1024):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
        neg_log_likelihood = outputs.loss
    
    return math.exp(neg_log_likelihood.item())

def calculate_average_perplexity(dataset):
    perplexities = []
    for sample in tqdm(dataset, desc="Calculating Perplexity", unit="sample"):
        text = sample["text"]
        perplexity = calculate_perplexity(model, tokenizer, text)
        perplexities.append(perplexity)
    print(f"Average Perplexity: {np.average(perplexities)}")

def calculate_average_inference_time(dataset):
    times=[]
    for sample in tqdm(dataset, desc="Calculating Perplexity", unit="sample"):
        text = sample["text"]
        t1=time.time()
        _=calculate_perplexity(model, tokenizer, text)
        t2=time.time()
        times.append(t2-t1)
    print(f"Average inference time (3000 samples): {np.average(times)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float32)
model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        device_map="auto",
        quantization_config=quantization_config,
    )
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

save_path="GPT2_BnB_8bit.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "quantization_config": quantization_config, 
}, save_path)

sum=0
for name,param in model.state_dict().items():
    sum+=param.nelement() * param.element_size()
sum=sum/(1024*1024)
print(f"Size of Quantized Model: {sum}")

dataset = load_dataset('wikipedia', '20220301.en', split='train[:3000]')

calculate_average_perplexity(dataset)
calculate_average_inference_time(dataset)