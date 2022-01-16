# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

from warnings import warn
import torch


class fftp:
    def __init__(self, dtype=float):
        pass
    def __call__(self,x, outn=None, ref=False):
        return torch.fft.fft(x)
class ifftp:
    def __init__(self, dtype=float):
        pass
    def __call__(self,x, outn=None, n=None, ref=False):
        return torch.fft.ifft(x,n=n)
class rfftp:
    def __init__(self, dtype=float):
        pass
    def __call__(self,x, outn=None, ref=False):
        return torch.fft.rfft(x)
class irfftp:
    def __init__(self, dtype=float):
        pass
    def __call__(self,x,outn=None,ref=False):
        return torch.fft.irfft(x,n=outn)
