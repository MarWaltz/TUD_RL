import torch

K = 3
batch_size = 1
num_actions = 4

x = []

for _ in range(K):
    x.append(torch.randn_like(torch.zeros(batch_size, num_actions)))

y = torch.stack(x, dim=0)

print(y)
print(y.shape)

from collections import Counter

a = torch.argmax(y, dim=2)

print(a)
print(torch.mode(a.flatten()).values.item())

print("\n")
t = torch.randn((1, num_actions))
print(t)
print(torch.argmax(t))

