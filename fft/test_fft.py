import torch

torch.ops.load_library("./build/libfft.so")

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.manual_seed(1)

def TestRfft():
    x0 = torch.rand(320)
    y0 = torch.fft.rfft(x0)
    y1 = torch.ops.top.rfft(x0, True)
    d = (y0 - y1).abs().max().item()

    print(f"Rfft")
    print(f"ref : {y0.shape}, {y0.abs().max()}, {y0.abs().min()}")
    print(f"cpp : {y1.shape}, {y1.abs().max()}, {y1.abs().min()}")
    print(f"diff: {d}")
    print()


def TestIrfft():
    xr = torch.rand(161)
    xi = torch.rand(161)
    x0 = torch.complex(xr, xi).contiguous()

    y0 = torch.fft.irfft(x0)
    y1 = torch.ops.top.irfft(x0, True)
    d = (y0 - y1).abs().max().item()

    print(f"Irfft")
    print(f"ref : {y0.shape}, {y0.abs().max()}, {y0.abs().min()}")
    print(f"cpp : {y1.shape}, {y1.abs().max()}, {y1.abs().min()}")
    print(f"diff: {d}")
    print()


if __name__ == "__main__":
    TestRfft()
    TestIrfft()
