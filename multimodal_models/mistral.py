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
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from transformers import Mistral3ForConditionalGeneration

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt_ttt, prompt_mnist

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('mistralai/Mistral-Small-3.2-24B-Instruct-2506', f'{cache_dir}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/95a6d26c4bfb886c58daf9d3f7332c857cb27b43'),
    ('mistralai/Mistral-Small-3.1-24B-Instruct-2503', f'{cache_dir}/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a')
]

with open(base_dir / Path('_dataset') / Path('dataset_img') / Path('data.json'), 'r') as f: data = json.load(f)
df = pd.DataFrame.from_dict(data, orient='index').reset_index()

for model_id in model_ids:

    output_file = base_dir.parent / Path(f'results/multimodal/{model_id[0].split("/")[1]}.npy')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(model_id[1]) / Path('tekken.json')
    tokenizer = MistralTokenizer.from_file(
        tokenizer_path
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=True,
        local_files_only=True,
        device_map='auto',
        dtype=torch.bfloat16
    )
    model.eval()
    
    eos_token = [2]
    
    results = []
    for idx, row in df.iterrows():

        prompt = prompt_ttt if idx < 50 else prompt_mnist
        image = Image.open(base_dir / Path(f'_dataset/dataset_img/{row["name"]}.png')).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt.strip()}, {'type': 'image','image': image}]}]
        tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
        input_ids = torch.tensor([tokenized.tokens]).to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        pixel_values = torch.tensor(tokenized.images[0], dtype=torch.bfloat16).unsqueeze(0).to(model.device)
        image_sizes = torch.tensor([pixel_values.shape[-2:]]).to(model.device)
        target_ids = ChatCompletionRequest(messages=[UserMessage(role='user', content=row['board'])])
        target_ids = tokenizer.encode_chat_completion(target_ids).tokens[2:-1] + eos_token 
        
        prob_hist = []
        log_probs = []
        top_tokens_hist = []
        top_probs_hist = []
        for token in target_ids:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, image_sizes=image_sizes)
            logits = outputs[0][:, -1, :].to(torch.float64)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs[0], k=20)
            top_tokens_hist.append(top_ids.tolist())
            top_probs_hist.append(top_probs.tolist())
            token_prob = probs[0, token].item()
            prob_hist.append(np.float64(token_prob))
            log_probs.append(np.log(np.float64(token_prob)))
            token_tensor = torch.tensor([[token]], device=input_ids.device)
            input_ids = torch.cat([input_ids, token_tensor], dim=1)
            attention_mask = torch.ones_like(input_ids).to(model.device)

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

    del model, tokenizer, input_ids, outputs, logits, probs, token_tensor, pixel_values, image_sizes
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()