import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import re
import gc
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed
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
    reasoning = True

    eos_token = tokenizer.encode('<｜end▁of▁sentence｜>')

    results = []
    truncated = 0
    truncated_ablation = 0
    for thinking_allowed in [True, False]:
        for _, row in df.iterrows():
            print(row)
            messages = [{'role': 'user', 'content': f'{prompt}\n{row["user"]}'}]
            target_ids = tokenizer.encode(row['target']) + eos_token[-1:]

            if thinking_allowed:
                input_data = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
                outputs = model.generate(**input_data, max_new_tokens=2048)
                text = tokenizer.decode(outputs[0])
                if row['ablation']:
                    if '</think>' not in text: truncated_ablation += 1
                else:
                    if '</think>' not in text: truncated += 1
                think_content = re.search(r'(<think>.*?</think>)', text + '</think>', re.DOTALL)
                think_content = think_content.group(1)
                trace_len = len(tokenizer.encode(think_content, add_special_tokens=False))
                
                token_tensor = torch.tensor([tokenizer.encode(think_content + '\n\n')], device=input_data['input_ids'].device, dtype=input_data['input_ids'].dtype)
                input_ids = torch.cat([input_data['input_ids'], token_tensor], dim=1)
                input_data['input_ids'] = input_ids
                input_data['attention_mask'] = torch.ones_like(input_data['input_ids'])
            else:
                trace_len = None
                input_data = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device=model.device)
                token_tensor = torch.tensor([tokenizer.encode('<think></think>\n\n')], device=input_data['input_ids'].device, dtype=input_data['input_ids'].dtype)
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
                top_probs, top_ids = torch.topk(probs[0], k=40)
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

    del model, tokenizer, prob_hist, log_probs, logits, token_tensor, target_ids, input_data, think_content, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()