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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()



sigma = 0.001
alpha = sigma / 2


MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DTYPE = torch.bfloat16  # use bf16 if your GPU supports it; else torch.float16

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1024}
num_iterations = 100
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
    # 1) Init engine on GPU 0, offline (no HTTP)
    for iteration in range(num_iterations):
        engine = sgl.Engine(
            model_path=MODEL,
            random_seed=42,
            base_gpu_id=0,
            # If your sglang version exposes kwargs, these mirror the envs:
            # use_cuda_graph=False,
            # attention_backend="torch",
            mem_fraction_static=0.7,
        )

        # 2) Build a CPU copy and zero its weights
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=DTYPE, device_map="cpu", trust_remote_code=True
        )
        total_species = 2
        generator_seeds = []
        rewards = []
        advantages = []
        total_params = list(model.state_dict().items())
        
        for species_idx in range(total_species):
            seed = randint(0, 2 ** 30)
            
            gpu_generator = torch.Generator(device="cuda:0").manual_seed(seed)
            generator_seeds.append(seed)
            named = []
            for i, (name, t) in enumerate(total_params):
                
                noise_tensor = torch.randn(t.shape, device = "cuda:0", generator = gpu_generator, dtype = DTYPE)
                if i == len(total_params) - 1:
                    torch.save(noise_tensor, f"saves1_{species_idx}_{name}.pt")
                named.append((name, t.to("cuda:0", non_blocking=True).add_(sigma * noise_tensor)))
                
                if i % max(len(total_params) // 10, 1) == 0 or i == len(total_params) - 1:
                    engine.update_weights_from_tensor(named)
                    del noise_tensor
                    named.clear()
                    
                    clear_vram()
                
            outputs = engine.generate(prompts, sampling_params)
            correct_answers = 0
            for i in range(len(outputs)):
                output = outputs[i]
                print(output["text"])
                answer, penalty = find_boxed_x(output["text"])
                if answer is None:
                    answer = ""
                print(answers[i])
                print(answer)
                if answer.strip() == str(answers[i]).strip():
                    correct_answers += 1
            reward = correct_answers / len(outputs)
            rewards.append(reward)
        reward_mean = statistics.mean(rewards)
        reward_std = statistics.stdev(rewards)
        for element in rewards:
            advantages.append((element - reward_mean) / reward_std)
        engine.shutdown()
        
        generators = []
        for species_idx, (seed, advantage) in enumerate(zip(generator_seeds, advantages)):
            generators.append(torch.Generator(device="cuda:0").manual_seed(seed))
        clear_vram()
        model.to("cuda:0")
        total_params2 = list(model.state_dict().items())
        for i in range(len(total_params2)):
            assert total_params2[i][0] == total_params[i][0], f"{total_params2[i][0]}, {total_params[i][0]}"
        
        for i, (name, t) in enumerate(total_params2):
            noise_sum_accumulator = torch.zeros(t.shape, device = "cuda:0", dtype = DTYPE)
            for species_idx, (generator, advantage) in enumerate(zip(generators, advantages)):
                
                generated_noise = torch.randn(t.shape, device = "cuda:0", generator = generator, dtype = DTYPE)
                noise_sum_accumulator += generated_noise

                if i == len(total_params2) - 1:
                    torch.save(generated_noise, f"saves2_{species_idx}_{name}.pt")
            print(noise_sum_accumulator.device, alpha, total_species, t.device)
            t.add_(alpha / total_species + noise_sum_accumulator)
            if i % max(len(total_params2) // 10, 1) == 0 or i == len(total_params2) - 1:
                clear_vram()
        
        
        
        clear_vram()
        print("DONE")
        time.sleep(100)
        
if __name__ == "__main__":
    dataset_path = "demo_dataset.json"
    if torch.cuda.device_count() < 1:
        raise SystemExit("Need at least one CUDA device.")
    with torch.no_grad():
        main(dataset_path)
