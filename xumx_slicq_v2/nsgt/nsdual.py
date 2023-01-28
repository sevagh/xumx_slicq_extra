import numpy as np
import torch

from .util import chkM


def nsdual(g, wins, nn, M=None, device="cpu"):
    M = chkM(M,g)

    # Construct the diagonal of the frame operator matrix explicitly
    x = torch.zeros((nn,), dtype=float, device=torch.device(device))
    for gi,mii,sl in zip(g, M, wins):
        xa = torch.square(torch.fft.fftshift(gi))
        xa *= mii
        x[sl] += xa

    gd = [gi/torch.fft.ifftshift(x[wi]) for gi,wi in zip(g,wins)]
    return gd
