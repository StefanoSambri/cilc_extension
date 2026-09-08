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

from utility.utility_functions import set_seeds, cache_dir, seed
from utility.prompts import prompt_ttt, prompt_mnist

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('allenai/Molmo-7B-D-0924', f'{cache_dir}/models--allenai--Molmo-7B-D-0924/snapshots/cab33fb7f1a40091911f81165f8481920621948f'),
    ('allenai/Molmo-7B-O-0924', f'{cache_dir}/models--allenai--Molmo-7B-O-0924/snapshots/7a8c4bf80c839c243a6908c6ebbb0f1ee576d7ca')
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
        trust_remote_code=True,
        local_files_only=True,
        device_map='auto',
        dtype=torch.bfloat16
    )
    model.eval()
    reasoning = False
    thinking_allowed = None
    truncated_ttt = None
    truncated_mnist = None
    truncated_ablation_ttt = None
    truncated_ablation_mnist = None
    trace_len = None

    eos_token = '<|endoftext|>'
    
    results = []
    for idx, row in df.iterrows():

        prompt = prompt_ttt if idx < 100 else prompt_mnist
        image = Image.open(base_dir / Path(f'_dataset/dataset_img/{row["name"]}.png')).convert('RGB')
        inputs = processor.process(images=image, text=prompt)
        target_ids = tokenizer.encode(row['board'] + eos_token)
        input_data = {k: v.unsqueeze(0).to(model.device, dtype=model.dtype) if torch.is_floating_point(v) else v.unsqueeze(0).to(model.device) for k, v in inputs.items()}

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
            token_tensor = torch.tensor([[token]], device=device, dtype=input_data['input_ids'].dtype)
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
            'perplexity': np.float64(perplexity),
        }
        results.append(result)

    with output_file.open('wb') as f: np.save(f, results, allow_pickle=True)
    if reasoning: 
        with output_file.with_suffix('.json').open('w') as f: json.dump([{'n_truncated_ttt': truncated_ttt}, {'n_truncated_mnist': truncated_mnist}, {'n_truncated_ablation_ttt': truncated_ablation_ttt}, {'n_truncated_ablation_mnist': truncated_ablation_mnist}], f, indent=2)

    del model, processor, tokenizer, logits, token_tensor, input_data, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()