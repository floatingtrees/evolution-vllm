import torch

y = torch.load("saves1_0_lm_head.weight.pt")
x = torch.load("saves2_0_lm_head.weight.pt")
print(y - x)