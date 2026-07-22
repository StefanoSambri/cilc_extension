import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import gc
import re
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt_ttt, prompt_mnist

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('AIDC-AI/Ovis2.5-2B', f'{cache_dir}/models--AIDC-AI--Ovis2.5-2B/snapshots/393c932b2a03e28eb9aaa503e3c4ab3ad384d958'),
    ('AIDC-AI/Ovis2.5-9B', f'{cache_dir}/models--AIDC-AI--Ovis2.5-9B/snapshots/d73b2283ae2a930b7762f8d7b8b8a3f0f3b5c3bd')
]

with open(base_dir / Path('_dataset') / Path('dataset_img') / Path('data.json'), 'r') as f: data = json.load(f)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()

for model_id in model_ids:

    output_file = base_dir.parent / Path(f'results/multimodal/{model_id[0].split("/")[1]}.npy')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extra_kwargs = {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=True,
        local_files_only=True,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation = 'flash_attention_2'
    )
    tokenizer = model.text_tokenizer
    
    model.eval()

    eos_token = '<|im_end|>'
      
    results = []
    for idx, row in df.iterrows():

        prompt = prompt_ttt if idx < 50 else prompt_mnist

        image = Image.open(base_dir / Path(f'_dataset/dataset_img/{row["name"]}.png')).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image','image': image}, {'type': 'text', 'text': prompt.strip()}]}]
        input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages=messages, add_generation_prompt=True, enable_thinking=True)
        input_ids = input_ids.to(device=model.device)
        pixel_values = pixel_values.to(device=model.device) if pixel_values is not None else None
        grid_thws = grid_thws.to(device=model.device) if grid_thws is not None else None
        target_ids = tokenizer.encode(row['board'] + eos_token)

        outputs = model.generate(inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws, enable_thinking=True, enable_thinking_budget=True, max_new_tokens=4096, thinking_budget=2048)
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        think_content = re.search(r'(<think>.*?</think>)', text + '</think>', re.DOTALL)
        think_content = think_content.group(1)
        token_tensor = torch.tensor([tokenizer.encode(think_content)], device=model.device)
        input_ids = torch.cat([input_ids, token_tensor], dim=1)
        attention_mask = torch.ones_like(input_ids)
                    
        prob_hist = []
        log_probs = []
        top_tokens_hist = []
        top_probs_hist = []
        for token in target_ids:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws)
            logits = outputs[0][:, -1, :].to(torch.float64)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs[0], k=20)
            top_tokens_hist.append(top_ids.tolist())
            top_probs_hist.append(top_probs.tolist())
            token_prob = probs[0, token].item()
            prob_hist.append(np.float64(token_prob))
            log_probs.append(np.log(np.float64(token_prob)))
            token_tensor = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, token_tensor], dim=1)
            attention_mask = torch.ones_like(input_ids).to(model.device)

        nll = -np.sum(np.array(log_probs, dtype=np.float64)) / np.float64(len(log_probs))
        perplexity = np.exp(nll).astype(np.float64)
        perplexity_99 = trimmed_perplexity(log_probs, 0.01)
        perplexity_95 = trimmed_perplexity(log_probs, 0.05)
        perplexity_90 = trimmed_perplexity(log_probs, 0.10)

        result = {
            'image_name': row['name'],
            'prompt': prompt,
            'target': row['board'],
            'target_ids': target_ids,
            'probabilities': np.array(prob_hist, dtype=np.float64),
            'top_20_tokens': top_tokens_hist,
            'top_20_probs': np.array(top_probs_hist, dtype=np.float64),
            'perplexity': np.float64(perplexity),
            'perplexity_99': np.float64(perplexity_99),
            'perplexity_95': np.float64(perplexity_95),
            'perplexity_90': np.float64(perplexity_90)
        }
        results.append(result)

    with output_file.open('wb') as f: np.save(f, results, allow_pickle=True)

    del model, tokenizer, input_ids, attention_mask, outputs, logits, probs, token_tensor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()