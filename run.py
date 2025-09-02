import torch

x = torch.randn((32, 16384)).cuda()
y = torch.randn((32, 16384)).cuda()
while True:
    z = torch.matmul(x, y.permute(1, 0))