import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import gc
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt_ttt, prompt_mnist

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('google/gemma-3-4b-it', f'{cache_dir}/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767'),
    ('google/gemma-3-12b-it', f'{cache_dir}/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80'),
    ('google/gemma-3-27b-it', f'{cache_dir}/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a')
]

with open(base_dir / Path('_dataset') / Path('dataset_img') / Path('data.json'), 'r') as f: data = json.load(f)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()

for model_id in model_ids:

    output_file = base_dir.parent / Path(f'results/multimodal/{model_id[0].split("/")[1]}.npy')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=True,
        local_files_only=True,  
        use_fast=False
    )
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=False,
        local_files_only=True,
        device_map='auto',
        dtype=torch.bfloat16
    )
    model.eval()

    eos_token = '<end_of_turn>'
    
    results = []
    for idx, row in df.iterrows():

        prompt = prompt_ttt if idx < 50 else prompt_mnist
        image = Image.open(base_dir / Path(f'_dataset/dataset_img/{row["name"]}.png')).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image','image': image}, {'type': 'text', 'text': prompt.strip()}]}]
        prefill_data = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device=model.device)
        prefill_tensor = torch.cat([prefill_data['input_ids'], torch.tensor([[2]], device=prefill_data['input_ids'].device, dtype=prefill_data['input_ids'].dtype)], dim=1)
        input_data = {k: v.clone() for k, v in prefill_data.items()}
        input_data['input_ids'] = prefill_tensor
        input_data['attention_mask'] = torch.ones_like(input_data['input_ids'])
        token_type_tensor = torch.tensor([[0]], device=prefill_tensor.device, dtype=input_data['token_type_ids'].dtype)
        input_data['token_type_ids'] = torch.cat([input_data['token_type_ids'], token_type_tensor], dim=1)
        target_ids = tokenizer.encode(row['board'] + eos_token)[1:]

        prob_hist = []
        log_probs = []
        top_tokens_hist = []
        top_probs_hist = []
        for token in target_ids:
            with torch.no_grad():
                outputs = model(**input_data)
            logits = outputs.logits[:, -1, :]
            logits = logits.to(torch.float64)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs[0], k=20)
            top_tokens_hist.append(top_ids.tolist())
            top_probs_hist.append(top_probs.tolist())
            token_prob = probs[0, token].item()
            prob_hist.append(np.float64(token_prob))
            log_probs.append(np.log(np.float64(token_prob)))
            token_tensor = torch.tensor([[token]], device=prefill_tensor.device, dtype=prefill_tensor.dtype)
            input_ids = torch.cat([input_data['input_ids'], token_tensor], dim=1)
            input_data['input_ids'] = input_ids
            input_data['attention_mask'] = torch.ones_like(input_data['input_ids']).to(model.device)
            token_type_tensor = torch.tensor([[1]], device=prefill_tensor.device, dtype=input_data['token_type_ids'].dtype)
            input_data['token_type_ids'] = torch.cat([input_data['token_type_ids'], token_type_tensor], dim=1)

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

    del model, processor, tokenizer, logits, token_tensor, input_data, outputs, prefill_tensor, token_type_tensor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()