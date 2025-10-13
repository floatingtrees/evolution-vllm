# swap_weights_sglang_safe.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sglang as sgl
import gc
import json
import re
from random import randint
import time
import sys
import statistics

from typing import Optional, Tuple

def find_boxed_x(string: str) -> Tuple[Optional[str], int]:
    # Match \boxed{ ... } with an integer that may contain commas, e.g. -1,234,567
    matches = re.findall(r'\\boxed\{\s*(-?\d{1,3}(?:,\d{3})*|\-?\d+)\s*\}', string)
    # Strip commas from each match
    cleaned = [m.replace(",", "") for m in matches]
    x = cleaned[-1] if cleaned else None
    penalty = 2 if len(matches) >= 2 else 1
    return x, penalty


def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()



sigma = 0.001
alpha = sigma / 2
VERIFY_DETERMINISM = False

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DTYPE = torch.bfloat16  # use bf16 if your GPU supports it; else torch.float16

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1024}
def update_weights(model, total_params, generators, advantages, total_species):
    model.to("cuda:0")
    clear_vram()
    total_params2 = list(model.state_dict().items())
    for i in range(len(total_params2)):
        assert total_params2[i][0] == total_params[i][0], f"{total_params2[i][0]}, {total_params[i][0]}"
        assert total_params2[i][1].shape == total_params[i][1].shape, f"{total_params2[i][1].shape}, {total_params[i][1].shape}"
    
    for i, (name, t) in enumerate(total_params2):
        noise_sum_accumulator = torch.zeros(t.shape, device = "cuda:0", dtype = DTYPE)
        noise_buffer = torch.zeros(t.shape, device = "cuda:0", dtype = DTYPE)
        for species_idx, (generator, advantage) in enumerate(zip(generators, advantages)):
            noise_buffer.normal_(mean = 0.0, std = 1.0, generator = generator) # Reuse the buffer to reduce VRAM consumption
            noise_sum_accumulator += noise_buffer * advantage

            if i == len(total_params2) - 1 and VERIFY_DETERMINISM:
                torch.save(noise_buffer, f"saves2_{species_idx}_{name}.pt")
        t.add_(alpha / total_species * noise_sum_accumulator)
        if i % max(len(total_params2) // 10, 1) == 0 or i == len(total_params2) - 1:
            del noise_sum_accumulator
            del noise_buffer
            clear_vram()
    model.cpu()
    del generators

def main(dataset_path):
    with open(dataset_path, "r") as f:
        datalist = json.load(f)
    prompts = []
    answers = []
    
    for i in range(len(datalist)):
        user_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": datalist[i]["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(user_input)
        answers.append(datalist[i]["answer"])
        
    clear_vram()
    # Initialize sglang engine
    engine = sgl.Engine(
        model_path="/mnt/t9/evolution200",
        random_seed=42,
        base_gpu_id=0,
        mem_fraction_static=0.75,
    )
    clear_vram()
    outputs = engine.generate(prompts, sampling_params)
    correct_answers = 0
    for i in range(len(outputs)):
        output = outputs[i]
        
        answer, penalty = find_boxed_x(output["text"])
        if answer is None:
            answer = ""
        
        if answer.strip() == str(answers[i]).strip():
            correct_answers += 1
        if i == len(outputs) - 1:
            print(output["text"])
            print(answers[i])
            print(answer)
    reward = correct_answers / len(outputs)
    print(reward)
if __name__ == "__main__":
    dataset_path = "data/demo_dataset_smol.json"
    if torch.cuda.device_count() < 1:
        raise SystemExit("Need at least one CUDA device.")
    with torch.no_grad():
        main(dataset_path)
