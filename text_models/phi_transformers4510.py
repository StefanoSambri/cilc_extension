import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import gc
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed
from utility.prompts import prompt

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('microsoft/Phi-4-mini-instruct', f'{cache_dir}/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083')
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
        attn_implementation='flash_attention_2',
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    model.eval()
    reasoning = False
    thinking_allowed = None
    truncated = None
    truncated_ablation = None
    trace_len = None

    eos_token = '<|end|>'

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

        result = {
            'reasoning': reasoning,
            'ablation' : row['ablation'],
            'thinking_allowed' : thinking_allowed,
            'trace_len' : trace_len,
            'prompt': row['user'],
            'target': row['target'],
            'target_ids': target_ids,
            'probabilities': np.array(prob_hist, dtype=np.float64),
            'top_40_tokens': top_tokens_hist,
            'top_40_probs': np.array(top_probs_hist, dtype=np.float64),
            'perplexity': np.float64(perplexity),
        }
        results.append(result)

    with output_file.open('wb') as f: np.save(f, results, allow_pickle=True) 
    if reasoning: 
        with output_file.with_suffix('.json').open('w') as f: json.dump([{'n_truncated': truncated}, {'n_truncated_ablation': truncated_ablation}], f, indent=2)   
        
    del model, tokenizer, prob_hist, log_probs, logits, token_tensor, target_ids, input_data, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()