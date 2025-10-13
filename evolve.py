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
from reward import compute_reward
from config_parser import load_config
from pathlib import Path

conf = load_config("conf/config.yaml")
sigma = conf.hyperparams.sigma
alpha = sigma / 2
model_save_path = conf.saves.save_path
LOG_PATH = conf.logging.logs_save_path
MODEL = conf.model
engine_mem_fraction = conf.sampling.engine_mem_fraction
sampling_params = {"temperature": conf.sampling.temperature, "top_p": conf.sampling.top_p, 
                   "max_new_tokens": conf.sampling.max_new_tokens}
num_iterations = conf.hyperparams.epochs
total_species = conf.hyperparams.total_species

VERIFY_DETERMINISM = False
if conf.dtype != "bfloat16":
    raise ValueError("Only currently supported dtype is bfloat16")
dataset_path = conf.data_path
DTYPE = torch.bfloat16

from typing import Optional, Tuple



def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    
def ensure_dir_recursive(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)




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
                torch.save(noise_buffer, f"tests/saves2_{species_idx}_{name}.pt")
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
        
    model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=DTYPE, device_map="cpu", trust_remote_code=True
        )
    for iteration in range(num_iterations):
        clear_vram()
        # Initialize sglang engine
        engine = sgl.Engine(
            model_path=MODEL,
            random_seed=42,
            base_gpu_id=0,
            mem_fraction_static=engine_mem_fraction,
        )
        total_params = list(model.state_dict().items())
        named = []
        # Write model weights from cpu copy into inference engine
        for i, (name, t) in enumerate(total_params):
            named.append((name, t.to("cuda:0", non_blocking=True)))
            
            if i % max(len(total_params) // 10, 1) == 0 or i == len(total_params) - 1:
                engine.update_weights_from_tensor(named)
                named.clear()
                
                clear_vram()
        
        outputs = engine.generate(prompts, sampling_params)
        rewards = []
        base_model_reward_list = []
        for i in range(len(outputs)):
            output = outputs[i]["text"]
            sample_reward = compute_reward(output, answers[i], LOG_PATH)
            base_model_reward_list.append(sample_reward)
        base_model_reward = statistics.mean(base_model_reward_list)
        print("Mean Reward without noise: ", round(base_model_reward, 3))
        generator_seeds = []
        rewards = []
        advantages = []
        
        
        for species_idx in range(total_species):
            clear_vram()
            seed = randint(0, 2 ** 30)
            
            gpu_generator = torch.Generator(device="cuda:0").manual_seed(seed)
            generator_seeds.append(seed)
            named = []
            for i, (name, t) in enumerate(total_params):
                
                noise_tensor = torch.randn(t.shape, device = "cuda:0", generator = gpu_generator, dtype = DTYPE)
                if i == len(total_params) - 1 and VERIFY_DETERMINISM:
                    torch.save(noise_tensor, f"saves1_{species_idx}_{name}.pt")
                named.append((name, t.to("cuda:0", non_blocking=True).add_(sigma * noise_tensor)))
                
                if i % max(len(total_params) // 10, 1) == 0 or i == len(total_params) - 1:
                    engine.update_weights_from_tensor(named)
                    del noise_tensor
                    named.clear()
                    
                    clear_vram()
                
            outputs = engine.generate(prompts, sampling_params)
            noise_model_reward_list = []
            for i in range(len(outputs)):
                output = outputs[i]["text"]
                sample_reward = compute_reward(output, answers[i], LOG_PATH)
                noise_model_reward_list.append(sample_reward)
            noise_model_reward = statistics.mean(noise_model_reward_list)
            rewards.append(noise_model_reward)
        reward_mean = statistics.mean(rewards)
        reward_std = statistics.stdev(rewards) + 1e-8
        print("Mean reward after noise: ", round(reward_mean, 3))
        sys.stdout.flush()
        for element in rewards:
            advantages.append((element - reward_mean) / reward_std)
        engine.shutdown()
        del engine
        
        generators = []
        for species_idx, (seed, advantage) in enumerate(zip(generator_seeds, advantages)):
            generators.append(torch.Generator(device="cuda:0").manual_seed(seed))
        clear_vram()
        update_weights(model, total_params, generators, advantages, total_species)
        if iteration % 50 == 0:
            ensure_dir_recursive(model_save_path)
            save_dir = f"{model_save_path}/model{iteration}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        
if __name__ == "__main__":
    if torch.cuda.device_count() < 1:
        raise SystemExit("Need at least one CUDA device.")
    with torch.no_grad():
        main(dataset_path)
