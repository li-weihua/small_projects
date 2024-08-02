import torch

torch.ops.load_library("./build/libfft.so")

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.manual_seed(1)

x0 = torch.rand(320)
y0 = torch.fft.rfft(x0)
y1 = torch.ops.top.rfft(x0)
d = (y0 - y1).abs().max().item()

print(f"ref : {y0.shape}, {y0.abs().max()}, {y0.abs().min()}")
print(f"cpp : {y1.shape}, {y1.abs().max()}, {y1.abs().min()}")
print(f"diff: {d}")
