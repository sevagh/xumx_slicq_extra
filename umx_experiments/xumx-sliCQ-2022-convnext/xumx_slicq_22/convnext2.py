'''
copied ConvNeXt architecture from 
https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/models/convnext.py

https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
'''

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from .convnext import LayerNorm, Block


class ConvNeXt2(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
             depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
             layer_scale_init_value=1e-6, head_init_scale=1.,
         ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=(5, 17), stride=(1, 4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=(3, 11), stride=(1, 2), dilation=(1, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.upsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(3):
            upsample_layer = nn.Sequential(
                LayerNorm(dims[-1-i], eps=1e-6, data_format="channels_first"),
                nn.ConvTranspose2d(dims[-1-i], dims[-2-i], kernel_size=(3, 11), stride=(1, 2), dilation=(1, 2), output_padding=(0, 1)),
            )
            self.upsample_layers.append(upsample_layer)
        stem = nn.Sequential(
            nn.ConvTranspose2d(dims[0], in_chans, kernel_size=(5, 21), stride=(1, 4), output_padding=(0, 3)),
            LayerNorm(in_chans, eps=1e-6, data_format="channels_first")
        )
        self.upsample_layers.append(stem)

        #self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        #self.head.weight.data.mul_(head_init_scale)
        #self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mix = x.detach().clone()

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        #x = self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

        for i in range(4):
            x = self.upsample_layers[i](x)

        # crop to original size
        x = x[..., : mix.shape[-2], : mix.shape[-1]]
        return x*mix

    #def forward(self, x):
    #    x = self.forward_features(x)
    #    x = self.head(x)
    #    return x


# tiny and small
#model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

#model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

# large
#model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

# xlarge
#model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
