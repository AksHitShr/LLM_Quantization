import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import math
import time

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

def calculate_average_inference_time(dataset,time_dequant):
    times=[]
    for sample in tqdm(dataset, desc="Calculating Perplexity", unit="sample"):
        text = sample["text"]
        t1=time.time()
        p=calculate_perplexity(model, tokenizer, text)
        t2=time.time()
        times.append(t2-t1)
    print(f"Average inference time (3000 samples): {np.average(times)+time_dequant}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()
dataset = load_dataset('wikipedia', '20220301.en', split='train[:3000]')

target_blocks = [model.transformer.h[0], model.transformer.h[1]]
target_block_names=["model.transformer.h[0]", "model.transformer.h[1]"]

for param in model.parameters():
    param.requires_grad = False

sum=0
for param in model.parameters():
    sum+=param.nelement() * param.element_size()
sum=sum/(1024*1024)
print(f"Size of Non-Quantized Model: {sum}")

quantized_params = {}
quantization_coeffs = {}

for i,block in enumerate(target_blocks):
    for name, param in block.named_parameters():
        param_fp32 = param.data.float()
        max_val = param_fp32.max()
        scale = max_val / 127
        param_int8 = torch.round(param_fp32 / scale).clamp(-127, 127).to(torch.int8)
        quantized_params[target_block_names[i]+"."+name] = param_int8
        quantization_coeffs[target_block_names[i]+"."+name] = {"scale": scale}

for i,block in enumerate(target_blocks):
    for name, param in block.named_parameters():
        param.data = quantized_params[target_block_names[i]+"."+name].to(param.data.device)

save_path="GPT2_SelectiveQuant.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "quantization_config": quantization_coeffs,  # Save the quantization configuration
}, save_path)

sum=0
for param in model.parameters():
    sum+=param.nelement() * param.element_size()
sum=sum/(1024*1024)
print(f"Size of Fully Quantized Model: {sum}")

t1=time.time()
def dequantize_param(name):
    scale = quantization_coeffs[name]["scale"]
    return quantized_params[name].float() * scale

for i,block in enumerate(target_blocks):
    for name, param in block.named_parameters():
        param.data = dequantize_param(target_block_names[i]+"."+name).to(param.data.device)
time_dequant=time.time()-t1

calculate_average_perplexity(dataset)
calculate_average_inference_time(dataset,time_dequant)