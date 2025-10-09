# swap_weights_sglang_safe.py
import os
import torch
from transformers import AutoModelForCausalLM
import sglang as sgl
import gc

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

# --- harden runtime to avoid the segfault path ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SGLANG_DISABLE_CUDA_GRAPH"] = "1"     # turn off CUDA graphs
os.environ["SGLANG_ATTN_BACKEND"] = "torch"       # avoid flashinfer/triton path
# Optional: limit engine memory use (rest stays free)
os.environ["SGLANG_MEM_FRACTION"] = "0.8"

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DTYPE = torch.bfloat16  # use bf16 if your GPU supports it; else torch.float16

sampling_params = {"temperature": 0.8, "top_p": 0.95}
def main():
    # 1) Init engine on GPU 0, offline (no HTTP)
    engine = sgl.Engine(
        model_path=MODEL,
        random_seed=42,
        base_gpu_id=0,
        # If your sglang version exposes kwargs, these mirror the envs:
        # use_cuda_graph=False,
        # attention_backend="torch",
        mem_fraction_static=0.8,
    )

    # 2) Build a CPU copy and zero its weights
    cpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    with torch.no_grad():
        for p in cpu_model.parameters():
            p.zero_()

    prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    ]

    print(engine.generate(prompts, sampling_params))
    # 3) Push zeroed weights into the engine (expects CUDA tensors)
    
    for name, t in cpu_model.state_dict().items():
        named = []
        if t.dtype != DTYPE:
            t = t.to(DTYPE)
        named.append((name, t.to("cuda:0", non_blocking=True)))
        print(name)
        engine.update_weights_from_tensor(named)
        clear_vram()

    print("✅ Swapped weights (zeroed) into the SGLang engine on GPU0.")
    prompts = [
    "W",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    ]

    outputs = engine.generate(prompts, sampling_params)
    print(outputs)
    
if __name__ == "__main__":
    if torch.cuda.device_count() < 1:
        raise SystemExit("Need at least one CUDA device.")
    main()
