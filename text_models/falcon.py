import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import re
import gc
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('tiiuae/falcon-7b-instruct', f'{cache_dir}/models--tiiuae--falcon-7b-instruct/snapshots/8782b5c5d8c9290412416618f36a133653e85285'),
    ('tiiuae/Falcon-H1-Tiny-90M-Instruct', f'{cache_dir}/models--tiiuae--Falcon-H1-Tiny-90M-Instruct/snapshots/e6389502a0b12cd8da894b395ba5bf7436873b16'),
    ('tiiuae/Falcon-H1R-7B', f'{cache_dir}/models--tiiuae--Falcon-H1R-7B/snapshots/a6f74bf181389908efd6970878d7ee2b42f5d417'),
    ('tiiuae/Falcon3-1B-Instruct', f'{cache_dir}/models--tiiuae--Falcon3-1B-Instruct/snapshots/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf'),
    ('tiiuae/Falcon3-3B-Instruct', f'{cache_dir}/models--tiiuae--Falcon3-3B-Instruct/snapshots/411bb94318f94f7a5735b77109f456b1e74b42a1'),
    ('tiiuae/Falcon3-7B-Instruct', f'{cache_dir}/models--tiiuae--Falcon3-7B-Instruct/snapshots/1e57a0ecd176c7c139f289c60a74e57f887c3dfb'),
    ('tiiuae/Falcon3-10B-Instruct', f'{cache_dir}/models--tiiuae--Falcon3-10B-Instruct/snapshots/8799bc6aec0152757221dc6b272d824642db6202'),
    ('tiiuae/Falcon3-Mamba-7B-Instruct', f'{cache_dir}/models--tiiuae--Falcon3-Mamba-7B-Instruct/snapshots/79268d5c8e650ec0ec24aad2729bfc906f569580'),
    ('tiiuae/Falcon-E-1B-Instruct', f'{cache_dir}/models--tiiuae--Falcon-E-1B-Instruct/snapshots/d20167318084c330b205e221eceed19354197bdc'),
    ('tiiuae/Falcon-E-3B-Instruct', f'{cache_dir}/models--tiiuae--Falcon-E-3B-Instruct/snapshots/deeb1e557f4eee27f0ebb2389604b8a67c56a74b')
]

df = pd.read_json(base_dir / Path('_dataset') / Path('dataset_txt.json'))

for model_id in model_ids:

    output_file = base_dir.parent / Path(f'results/text/{model_id[0].split("/")[1]}.npy')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=False,
        local_files_only=True,  
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=False,
        local_files_only=True,
        device_map='auto',
        dtype=torch.bfloat16
    )

    model.eval()

    if re.search(r'-H1|Mamba|-E-', model_id[0]):
        eos_token = '<|im_end|>'
    else:
        eos_token = '<|endoftext|>'
    
    results = []
    for _, row in df.iterrows():
        messages = [{'role': 'user', 'content': f'{prompt}\n{row["user"]}'}]

        input_data = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
        target_ids = tokenizer.encode(row['target'] + eos_token)

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
            token_tensor = torch.tensor([[token]], device=input_data['input_ids'].device, dtype=input_data['input_ids'].dtype)
            input_ids = torch.cat([input_data['input_ids'], token_tensor], dim=1)
            input_data['input_ids'] = input_ids
            input_data['attention_mask'] = torch.ones_like(input_data['input_ids']).to(model.device)

        nll = -np.sum(np.array(log_probs, dtype=np.float64)) / np.float64(len(log_probs))
        perplexity = np.exp(nll).astype(np.float64)
        perplexity_99 = trimmed_perplexity(log_probs, 0.01)
        perplexity_95 = trimmed_perplexity(log_probs, 0.05)
        perplexity_90 = trimmed_perplexity(log_probs, 0.10)

        result = {
            'prompt': row['user'],
            'target': row['target'],
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

    del model, tokenizer, prob_hist, log_probs, logits, token_tensor, target_ids, input_data, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()