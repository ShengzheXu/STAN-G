import torch 

sz = 5
print(torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1))

print(torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=0))

print(1 - torch.triu(torch.ones(sz, sz), diagonal=0))
