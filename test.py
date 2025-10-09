import torch

g = torch.Generator(device='cuda')
g.manual_seed(1234)
torch.use_deterministic_algorithms(True) 

print(torch.randn((1, 2), device = "cuda", generator=g))