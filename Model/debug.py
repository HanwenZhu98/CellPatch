import torch
import torch.nn as nn
import math

gene2vector = nn.Embedding(70000, 256)

a = gene2vector(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
print(a.mean(1))
print(a.std(1))

count2vector = nn.Linear(1, 256)
b = count2vector(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).view(-1, 1).float())

print(b.mean(1))
print(b.std(1))

# count2vector_norm = nn.LayerNorm(256)
count2vector_norm = nn.BatchNorm1d(256)
c = count2vector_norm(b)
print(c.mean(0))
print(c.std(0))