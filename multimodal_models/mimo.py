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
from transformers import AutoProcessor, AutoModelForImageTextToText

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed
from utility.prompts import prompt_ttt, prompt_mnist

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('XiaomiMiMo/MiMo-VL-7B-RL-2508', f'{cache_dir}/models--XiaomiMiMo--MiMo-VL-7B-RL-2508/snapshots/4bfb270765825d2fa059011deb4c96fdd579be6f')
]

with open(base_dir / Path('_dataset') / Path('dataset_img') / Path('data.json'), 'r') as f: data = json.load(f)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()

for model_id in model_ids:

    output_file = base_dir.parent / Path(f'results/multimodal/{model_id[0].split("/")[1]}.npy')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    processor = AutoProcessor.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=False,
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
    reasoning = True

    eos_token = '<|endoftext|>'

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
            target_ids = tokenizer.encode(row['board'] + eos_token)

            if thinking_allowed:
                inputs = processor.apply_chat_template(messages, add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',).to(device=model.device)
                outputs = model.generate(**inputs, max_new_tokens=2048)
                text = processor.decode(outputs[0])
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
                input_data = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device=model.device)
                token_tensor = torch.tensor([tokenizer.encode(think_content)], device=input_data['input_ids'].device, dtype=input_data['input_ids'].dtype)
                input_ids = torch.cat([input_data['input_ids'], token_tensor], dim=1)
                input_data['input_ids'] = input_ids
                input_data['attention_mask'] = torch.ones_like(input_data['input_ids'])
            else:
                trace_len = None
                input_data = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device=model.device)
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
                logits = outputs[0][:, -1, :]
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

    del model, tokenizer, input_ids, outputs, logits, probs, token_tensor, think_content, text, input_data
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()