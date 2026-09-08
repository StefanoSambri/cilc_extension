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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLMoeForConditionalGeneration

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt_ttt, prompt_mnist

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('Qwen/Qwen2-VL-2B-Instruct', f'{cache_dir}/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c'),
    ('Qwen/Qwen2-VL-7B-Instruct', f'{cache_dir}/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac'),
    ('Qwen/Qwen2.5-VL-3B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3'),
    ('Qwen/Qwen2.5-VL-7B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'),
    ('Qwen/Qwen2.5-VL-32B-Instruct', f'{cache_dir}/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da'),
    ('Qwen/Qwen3-VL-2B-Instruct', f'{cache_dir}/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203'),
    ('Qwen/Qwen3-VL-4B-Instruct', f'{cache_dir}/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'),
    ('Qwen/Qwen3-VL-8B-Instruct', f'{cache_dir}/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
    ('Qwen/Qwen3-VL-32B-Instruct', f'{cache_dir}/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb'),
    ('Qwen/Qwen3-VL-30B-A3B-Instruct', f'{cache_dir}/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c')
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
    if re.search(r'Qwen2', model_id[0]):
        model = AutoModelForImageTextToText.from_pretrained(
            model_id[1], 
            cache_dir=cache_dir, 
            trust_remote_code=False,
            local_files_only=True,
            device_map='auto',
            dtype=torch.bfloat16
        )
    elif re.search(r'-A3B-', model_id[0]):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id[1], 
            cache_dir=cache_dir, 
            trust_remote_code=False,
            local_files_only=True,
            device_map='auto',
            dtype=torch.bfloat16
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id[1], 
            cache_dir=cache_dir, 
            trust_remote_code=False,
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

    eos_token = '<|im_end|>'
  
    results = []
    for idx, row in df.iterrows():
        prompt = prompt_ttt if idx < 100 else prompt_mnist
        image = Image.open(base_dir / Path(f'_dataset/dataset_img/{row["name"]}.png')).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image','image': image}, {'type': 'text', 'text': prompt.strip()}]}]
        input_data = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device=model.device)
        target_ids = tokenizer.encode(row['board'] + eos_token)

        prob_hist = []
        log_probs = []
        top_tokens_hist = []
        top_probs_hist = []
        for token in target_ids:
            with torch.no_grad():
                outputs = model(**input_data)
            logits = outputs.logits[:, -1, :].to(torch.float64)
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

    del model, processor, tokenizer, logits, token_tensor, input_data, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()