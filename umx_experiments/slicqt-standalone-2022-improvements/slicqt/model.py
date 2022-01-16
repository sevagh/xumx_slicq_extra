import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ReLU, Tanh, BatchNorm2d, Conv2d, ConvTranspose2d, Sigmoid, Linear, BatchNorm3d, ModuleList
import copy


class DeOverlapNet(nn.Module):
    def __init__(
        self,
        nsgt,
        slicq_sample_input,
    ):
        super(DeOverlapNet, self).__init__()

        self.nsgt = nsgt

        ola = self.nsgt.overlap_add(slicq_sample_input)

        # linear layers for the deoverlap
        deoverlap_layers = [None]*len(slicq_sample_input)
        deoverlap_wins = [None]*len(slicq_sample_input)

        for i, slicq_ in enumerate(slicq_sample_input):
            nb_m_bins = slicq_.shape[-1]
            nwin = nb_m_bins
            deoverlap_layers[i] = Linear(in_features=nwin, out_features=nwin, bias=True)
            deoverlap_wins[i] = nwin

        self.deoverlap_layers = ModuleList(deoverlap_layers)
        self.deoverlap_wins = deoverlap_wins

    def freeze(self):
        for p in self.parameters():
            p.grad = None
        self.eval()

    def forward(self, x: Tensor, nb_slices, ragged_shapes) -> Tensor:
        # input is the overlap-added sliCQT

        # deoverlap with linear layer
        x = self.deoverlap_add(x, nb_slices)

        return x

    def deoverlap_add(self, slicq, nb_slices):
        ret = [None]*len(slicq)
        for i, slicq_ in enumerate(slicq):
            nwin = self.deoverlap_wins[i]
            hop = nwin//2
            nb_m_bins = nwin

            nb_samples, nb_channels, nb_f_bins, ncoefs = slicq_.shape

            out = torch.zeros((nb_samples, nb_channels, nb_f_bins, nb_slices, nwin), dtype=slicq_.dtype, device=slicq_.device)

            # each slice considers nwin coefficients

            ptr = 0
            for j in range(nb_slices):
                # inverse of overlap-add
                out[:, :, :, j, :] = self.deoverlap_layers[i](slicq_[:, :, :, ptr:ptr+nwin])
                ptr += hop

            ret[i] = out

        return ret
