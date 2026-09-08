
import os

os.environ['OMP_NUM_THREADS'] = str(os.getenv('SLURM_CPUS_PER_TASK'))

import gc
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import openai_harmony as oh
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, Mxfp4Config

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))

from utility.utility_functions import set_seeds, get_kernel, cache_dir, seed, trimmed_perplexity
from utility.prompts import prompt

import kernels
kernels.get_kernel = get_kernel
import transformers.integrations.hub_kernels as hub_kernels
hub_kernels.get_kernel = get_kernel

set_seeds(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK'))) 

model_ids = [
    ('openai/gpt-oss-20b', f'{cache_dir}/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')
]

df = pd.read_json(base_dir / Path('_dataset') / Path('dataset_txt.json'))

for model_id in model_ids:

    output_file = base_dir.parent / Path(f'results/text/{model_id[0].split("/")[1]}.npy')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    generation_config = GenerationConfig.from_pretrained(
        model_id[1], 
        cache_dir = cache_dir,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=False,
        local_files_only=True,  
        use_fast=False
    )
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id[1], 
        cache_dir=cache_dir, 
        trust_remote_code=False,
        local_files_only=True,
        device_map='auto',
        dtype=torch.bfloat16,
        quantization_config = quantization_config
    )
    model.eval()
    reasoning = True

    encoding = oh.load_harmony_encoding(oh.HarmonyEncodingName.HARMONY_GPT_OSS)
    eos_token = encoding.stop_tokens_for_assistant_actions()     
    end_token_id = encoding.encode('<|end|>', allowed_special={'<|end|>'})[0]

    sysm_analysis = (
        oh.SystemContent.new()
        .with_reasoning_effort(getattr(oh.ReasoningEffort, 'MEDIUM'))
        .with_conversation_start_date(str(datetime.today().date()))
        .with_required_channels(['analysis'])
    )
    sysm_final = (
        oh.SystemContent.new()
        .with_reasoning_effort(getattr(oh.ReasoningEffort, 'MEDIUM'))
        .with_conversation_start_date(str(datetime.today().date()))
        .with_required_channels(['analysis', 'final'])
    )

    results = []
    truncated = 0
    truncated_ablation = 0 
    for thinking_allowed in [True, False]:
        for _, row in df.iterrows():

            target_ids = encoding.encode(row['target']) + eos_token[-1:]
            if thinking_allowed:
                conv_analysis = oh.Conversation.from_messages([
                    oh.Message.from_role_and_content(oh.Role.SYSTEM, sysm_analysis),
                    oh.Message.from_role_and_content(oh.Role.USER, f'{prompt}\n{row["user"]}')
                ])          
                ids_analysis = encoding.render_conversation_for_completion(conv_analysis, oh.Role.ASSISTANT)
                ids_analysis = torch.tensor(ids_analysis, device=model.device).unsqueeze(0)   
                with torch.no_grad():      
                    out_analysis = model.generate(input_ids=ids_analysis, max_new_tokens=2048, eos_token_id=eos_token, do_sample=False, use_cache=True)
                raw_analysis = out_analysis[0][len(ids_analysis[0]):]
                text_analysis = encoding.parse_messages_from_completion_tokens(raw_analysis, oh.Role.ASSISTANT)
                if row['ablation']:
                    if end_token_id not in raw_analysis: truncated_ablation +=1
                else:
                    if end_token_id not in raw_analysis: truncated +=1
                text_analysis = text_analysis[0].content[0].text if text_analysis and text_analysis[0].content else ''
                text_analysis = '<|channel|>analysis<|message|>' + text_analysis.replace('<|end|>', '')
                trace_len = len(encoding.encode(text_analysis.replace('<|channel|>analysis<|message|>', '').replace('<|end|>', '')))
                conv_final = oh.Conversation.from_messages([
                    oh.Message.from_role_and_content(oh.Role.SYSTEM, sysm_final),
                    oh.Message.from_role_and_content(oh.Role.USER, f'{prompt}\n{row["user"]}'),
                    oh.Message.from_role_and_content(oh.Role.ASSISTANT, text_analysis)
                ])
                ids_final = encoding.render_conversation_for_completion(conv_final, oh.Role.ASSISTANT)
                ids_final_suffix = encoding.encode('<|channel|>final<|message|>', allowed_special={'<|channel|>','<|message|>'})
                ids_final += ids_final_suffix
                input_tensor = torch.tensor(ids_final, device=model.device).unsqueeze(0)
            else:
                conv_analysis = oh.Conversation.from_messages([
                    oh.Message.from_role_and_content(oh.Role.SYSTEM, sysm_final),
                    oh.Message.from_role_and_content(oh.Role.USER, f'{prompt}\n{row["user"]}')
                ])
                trace_len = None
                ids_final = encoding.render_conversation_for_completion(conv_analysis, oh.Role.ASSISTANT)
                ids_final_suffix = encoding.encode('<|channel|>final<|message|>', allowed_special={'<|channel|>','<|message|>'})
                ids_final += ids_final_suffix
                input_tensor = torch.tensor(ids_final, device=model.device).unsqueeze(0)

            prob_hist = []
            log_probs = []
            top_tokens_hist = []
            top_probs_hist = []
            for token in target_ids:
                with torch.no_grad():  
                    outputs = model(input_ids=input_tensor)
                logits = outputs.logits[:, -1, :]
                logits = logits.to(torch.float64)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_probs, top_ids = torch.topk(probs[0], k=40)
                top_tokens_hist.append(top_ids.tolist())
                top_probs_hist.append(top_probs.tolist())
                token_prob = probs[0, token].item()
                prob_hist.append(np.float64(token_prob))
                log_probs.append(np.log(np.float64(token_prob)))
                token_tensor = torch.tensor([[token]], device=input_tensor.device, dtype=input_tensor.dtype)
                input_tensor = torch.cat([input_tensor, token_tensor], dim=1)
                
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
                'perplexity': np.float64(perplexity)
            }
            results.append(result)

    with output_file.open('wb') as f: np.save(f, results, allow_pickle=True)  
    if reasoning: 
        with output_file.with_suffix('.json').open('w') as f: json.dump([{'n_truncated': truncated}, {'n_truncated_ablation': truncated_ablation}], f, indent=2)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()