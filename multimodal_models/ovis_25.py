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

from utility.utility_functions import set_seeds, cache_dir, seed
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
    reasoning = True

    eos_token = '<|im_end|>'
      
    results = []
    truncated_ttt = 0
    truncated_mnist = 0
    truncated_ablation_ttt = 0
    truncated_ablation_mnist = 0
    for thinking_allowed in [True, False]:
        for idx, row in df.iterrows():

            prompt = prompt_ttt if idx < 100 else prompt_mnist
            image = Image.open(base_dir / Path(f'_dataset/dataset_img/{row["name"]}.png')).convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image','image': image}, {'type': 'text', 'text': prompt.strip()}]}]
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages=messages, add_generation_prompt=True, enable_thinking=True)
            input_ids = input_ids.to(device=model.device)
            pixel_values = pixel_values.to(device=model.device) if pixel_values is not None else None
            grid_thws = grid_thws.to(device=model.device) if grid_thws is not None else None
            attention_mask = torch.ones_like(input_ids)
            target_ids = tokenizer.encode(row['board'] + eos_token)

            if thinking_allowed:
                outputs = model.generate(inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws, enable_thinking=True, enable_thinking_budget=True, max_new_tokens=4096, thinking_budget=2048, do_sample=False)
                text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                if row['name'] == 'ablation_img' and idx < 100:
                    if '</think>' not in text: truncated_ablation_ttt += 1
                elif row['name'] == 'ablation_img' and idx >= 100:
                    if '</think>' not in text: truncated_ablation_mnist += 1
                elif idx < 100:
                    if '</think>' not in text: truncated_ttt += 1
                elif idx >= 100:
                    if '</think>' not in text: truncated_mnist += 1
                think_content = re.search(r'(<think>.*?</think>)', text + '</think>', re.DOTALL)
                think_content = think_content.group(1)
                trace_len = len(tokenizer.encode(think_content, add_special_tokens=False))
                token_tensor = torch.tensor([tokenizer.encode(think_content + '\n\n')], device=model.device)
                input_ids = torch.cat([input_ids, token_tensor], dim=1)
                attention_mask = torch.ones_like(input_ids)
            else:
                trace_len = None
                token_tensor = torch.tensor([tokenizer.encode('<think></think>\n\n')], device=model.device)
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
                top_probs, top_ids = torch.topk(probs[0], k=40)
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

            result = {
                'reasoning': reasoning,
                'thinking_allowed' : thinking_allowed,
                'trace_len' : trace_len,
                'image_name': row['name'],
                'prompt': prompt,
                'target': row['board'],
                'target_ids': target_ids,
                'probabilities': np.array(prob_hist, dtype=np.float64),
                'top_40_tokens': top_tokens_hist,
                'top_40_probs': np.array(top_probs_hist, dtype=np.float64),
                'perplexity': np.float64(perplexity)
            }
            results.append(result)

    with output_file.open('wb') as f: np.save(f, results, allow_pickle=True)
    if reasoning: 
        with output_file.with_suffix('.json').open('w') as f: json.dump([{'n_truncated_ttt': truncated_ttt}, {'n_truncated_mnist': truncated_mnist}, {'n_truncated_ablation_ttt': truncated_ablation_ttt}, {'n_truncated_ablation_mnist': truncated_ablation_mnist}], f, indent=2)

    del model, tokenizer, input_ids, attention_mask, outputs, logits, probs, token_tensor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()