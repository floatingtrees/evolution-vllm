import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


g = torch.Generator(device='cuda')
g.manual_seed(1234)
torch.use_deterministic_algorithms(True) 

print(torch.randn((1, 2), device = "cuda", generator=g))

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
for name, buf in model.named_buffers():
    print(name, tuple(buf.shape), buf.dtype)
    print(buf)

total_params = list(model.state_dict().items())
for name, t in total_params:
    print(name)