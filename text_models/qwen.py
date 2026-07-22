import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import re
import gc
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 


model_ids = [
    ('Qwen/Qwen2-0.5B-Instruct', f'{cache_dir}/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d'),
    ('Qwen/Qwen2-1.5B-Instruct', f'{cache_dir}/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8'),
    ('Qwen/Qwen2-7B-Instruct', f'{cache_dir}/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c'),
    ('Qwen/Qwen2.5-0.5B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'),
    ('Qwen/Qwen2.5-1.5B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
    ('Qwen/Qwen2.5-3B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1'),
    ('Qwen/Qwen2.5-7B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'),
    ('Qwen/Qwen2.5-14B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'),
    ('Qwen/Qwen2.5-32B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'),
    ('Qwen/Qwen3-0.6B', f'{cache_dir}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'),
    ('Qwen/Qwen3-1.7B', f'{cache_dir}/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    ('Qwen/Qwen3-4B', f'{cache_dir}/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'),
    ('Qwen/Qwen3-8B', f'{cache_dir}/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'),
    ('Qwen/Qwen3-14B', f'{cache_dir}/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18'),
    ('Qwen/Qwen3-32B', f'{cache_dir}/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137'),
    ('Qwen/Qwen3-4B-Instruct-2507', f'{cache_dir}/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'),
    ('Qwen/Qwen3-30B-A3B-Instruct-2507', f'{cache_dir}/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'),
    ('Qwen/QwQ-32B', f'{cache_dir}/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0')
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

    eos_token = '<|im_end|>'
  
    results = []
    for _, row in df.iterrows():
        messages = [{'role': 'user', 'content': f'{prompt}\n{row["user"]}'}]
        target_ids = tokenizer.encode(row['target'] + eos_token)

        prob_hist = []
        log_probs = []
        top_tokens_hist = []
        top_probs_hist = []
        if re.search(r'Instruct', model_id[0]):
        
            input_data = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
            
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
        else:

            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',).to(device=model.device)
            outputs = model.generate(**inputs, max_new_tokens=2048)
            text = tokenizer.decode(outputs[0])
            think_content = re.search(r'(<think>.*?</think>)', text + '</think>', re.DOTALL)
            think_content = think_content.group(1)

            input_data = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device=model.device)
            prefill_tensor = input_data['input_ids']
            token_tensor = torch.tensor([tokenizer.encode(think_content)], device=prefill_tensor.device, dtype=prefill_tensor.dtype)
            input_ids = torch.cat([input_data['input_ids'], token_tensor], dim=1)
            input_data['input_ids'] = input_ids
            input_data['attention_mask'] = torch.ones_like(input_data['input_ids'])

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