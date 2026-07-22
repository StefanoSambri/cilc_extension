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
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', f'{cache_dir}/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'),
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', f'{cache_dir}/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60'),
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', f'{cache_dir}/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/1df8507178afcc1bef68cd8c393f61a886323761'),
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', f'{cache_dir}/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746')
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

    eos_token = tokenizer.encode('<｜end▁of▁sentence｜>')

    results = []
    for _, row in df.iterrows():

        messages = [{'role': 'user', 'content': f'{prompt}\n{row["user"]}'}]
        input_data = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
        outputs = model.generate(**input_data, max_new_tokens=2048)
        text = tokenizer.decode(outputs[0])
        think_content = re.search(r'(<think>.*?</think>)', text + '</think>', re.DOTALL)
        think_content = think_content.group(1)
    
        target_ids = tokenizer.encode(row['target']) + eos_token[-1:]

        prefill_tensor = input_data['input_ids']
        token_tensor = torch.tensor([tokenizer.encode(think_content)], device=prefill_tensor.device, dtype=prefill_tensor.dtype)
        input_ids = torch.cat([input_data['input_ids'], token_tensor], dim=1)
        input_data['input_ids'] = input_ids
        input_data['attention_mask'] = torch.ones_like(input_data['input_ids'])

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

    del model, tokenizer, prob_hist, log_probs, logits, token_tensor, target_ids, input_data, prefill_tensor, think_content, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()