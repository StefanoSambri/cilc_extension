import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import gc
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoTokenizer

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, load_image, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt_ttt, prompt_mnist

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('OpenGVLab/InternVL2-1B', f'{cache_dir}/models--OpenGVLab--InternVL2-1B/snapshots/0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187'),
    ('OpenGVLab/InternVL2-2B', f'{cache_dir}/models--OpenGVLab--InternVL2-2B/snapshots/e4f6747bd20f139e637642c6a058c6bd00b36919'),
    ('OpenGVLab/InternVL2-4B', f'{cache_dir}/models--OpenGVLab--InternVL2-4B/snapshots/6eb21ec278fbdb03b751a3d48d8dfc9724e05608'),
    ('OpenGVLab/InternVL2-8B', f'{cache_dir}/models--OpenGVLab--InternVL2-8B/snapshots/6fb9ad6924f69424e57fab2ab061d707688f0296'),
    ('OpenGVLab/InternVL2-26B', f'{cache_dir}/models--OpenGVLab--InternVL2-26B/snapshots/46f37972e05604cc44a59f72334df220fa197a7b'),
    ('OpenGVLab/InternVL2_5-1B', f'{cache_dir}/models--OpenGVLab--InternVL2_5-1B/snapshots/9d423ea1ae9f893897ee3f7493141073f5afcf22'),
    ('OpenGVLab/InternVL2_5-2B', f'{cache_dir}/models--OpenGVLab--InternVL2_5-2B/snapshots/573169ee54df216786bb9a189e9a32a060a008cf'),
    ('OpenGVLab/InternVL2_5-4B', f'{cache_dir}/models--OpenGVLab--InternVL2_5-4B/snapshots/2cf4a8158bbc40d35015e7c63b527890de4d27b3'),
    ('OpenGVLab/InternVL2_5-8B', f'{cache_dir}/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463'),
    ('OpenGVLab/InternVL2_5-26B', f'{cache_dir}/models--OpenGVLab--InternVL2_5-26B/snapshots/b537a9974b89cb621e9e6b9e7ebe2a904334fff0'),
    ('OpenGVLab/InternVL2_5-38B', f'{cache_dir}/models--OpenGVLab--InternVL2_5-38B/snapshots/32ba5a61b8a870d9ac946b639b134abed529d45c'),
    ('OpenGVLab/InternVL3-1B-Instruct', f'{cache_dir}/models--OpenGVLab--InternVL3-1B-Instruct/snapshots/f91c6391476dbc8fd07674d2589e8168d3ca667f'),
    ('OpenGVLab/InternVL3-2B-Instruct', f'{cache_dir}/models--OpenGVLab--InternVL3-2B-Instruct/snapshots/f6c7b60375759170fd49f5e9e298e2178485c5ba'),
    ('OpenGVLab/InternVL3-8B-Instruct', f'{cache_dir}/models--OpenGVLab--InternVL3-8B-Instruct/snapshots/ddb3a169d5582e5c76e0809a128e55ab63686ada'),
    ('OpenGVLab/InternVL3-14B-Instruct', f'{cache_dir}/models--OpenGVLab--InternVL3-14B-Instruct/snapshots/6ffe6f06d88b6e7e3bd6eccdf73028fae7855d38'),
    ('OpenGVLab/InternVL3-38B-Instruct', f'{cache_dir}/models--OpenGVLab--InternVL3-38B-Instruct/snapshots/150ad666e0c733b809742decded1f7484995b99c'),
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=True,
        local_files_only=True,  
        use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=True,
        local_files_only=True,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.eval()

    eos_token = '<|im_end|>'

    results = []
    for idx, row in df.iterrows():

        prompt = prompt_ttt if idx < 50 else prompt_mnist
        pixel_values = load_image(base_dir / Path(f'_dataset/dataset_img/{row["name"]}.png'), max_num=12).to(model.device, dtype=model.dtype)
        image_tokens = '<IMG_CONTEXT>' * (model.num_image_token * pixel_values.shape[0])
        text_with_img_pad = f'{image_tokens}\n{prompt}'
        tokenizer.padding_side = 'left'
        prefill_data = tokenizer(text_with_img_pad, return_tensors='pt', padding=True)
        image_flags = torch.ones((pixel_values.shape[0], model.num_image_token), dtype=torch.long, device=pixel_values.device)
        input_data = {k: v.clone().to(device=model.device) for k, v in prefill_data.items()}
        input_data['input_ids'] = input_data['input_ids'].long()
        target_ids = tokenizer.encode(row['board'] + eos_token)

        prob_hist = []
        log_probs = []
        top_tokens_hist = []
        top_probs_hist = []
        for token in target_ids:
            with torch.no_grad():
                outputs = model(input_ids=input_data['input_ids'], pixel_values=pixel_values, attention_mask=input_data['attention_mask'], image_flags=image_flags)
            logits = outputs[0][:, -1, :].to(torch.float64)
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

    del model, processor, input_ids, outputs, logits, probs, token_tensor, input_data, image_flags, pixel_values
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()