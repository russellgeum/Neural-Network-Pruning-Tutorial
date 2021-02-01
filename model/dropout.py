import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# By Shim Kyung Hwan
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# class SortDropout_v2 (nn.Module):

#     def __init__ (self, drop_ratio):
#         super(SortDropout_v2, self).__init__()
#         self.drop_ratio = drop_ratio
#         self.eps        = 1e-8

#     def forward (self, inputs):
#         mask    = torch.ones_like(inputs).type_as(inputs).to(inputs.device)
#         maskInd = int(math.floor(inputs.size(1)*(1-self.drop_ratio)))

#         maskInd = maskInd if maskInd > 0 else 1
#         mask[ :, maskInd:, :, :] = 0.0
#         return inputs * mask


# class SortDropout_v1 (nn.Module):

#     def __init__(self, drop_ratio):
#         super(SortDropout_v1, self).__init__()
#         self.drop_ratio = drop_ratio
#         self.eps        = 1e-8

#     def compute_sort_mask (self, inputs):
#         '''
#         Tensor shape is [out_channels, in_channels, height, width]
#         So, [1, in_channels, height, width] size is feature map of single batch data
#         '''
#         out_channel, in_channel, height, width = inputs.shape
#         mask  = torch.ones_like(inputs)
#         index = np.arange(in_channel)
#         dead  = index[int((1-self.drop_ratio) * in_channel): ]
#         mask[ :, dead, :, :] = 0

#         return mask
        
#     def forward (self, inputs):

#         if self.drop_ratio == 0.0:
#             return inputs
#         else:
#             mask = self.compute_sort_mask(inputs)
#             return mask * inputs


# class Dropout2d (nn.Module):

#     def __init__ (self, drop_ratio = 0.0):
#         super(Dropout2d, self).__init__()
#         self.drop_ratio = drop_ratio
    
#     def forward (self, inputs):

#         if self.drop_ratio == 0.0:
#             return inputs
#         elif self.drop_ratio > 0.0 and self.drop_ratio < 1.0:
#             return nn.Dropout2d(self.drop_ratio)(inputs)


# class DropBlock2d (nn.Module):
#     """
#     Randomly zeroes 2D spatial blocks of the input tensor.
#     As described in the paper
#     `DropBlock: A regularization method for convolutional networks`_ ,
#     dropping whole blocks of feature map allows to remove semantic
#     information as compared to regular dropout.
#     Args:
#         drop_ratio (float): probability of an element to be dropped.
#         block_size (int): size of the block to drop
#     Shape:
#         - Input: `(N, C, H, W)`
#         - Output: `(N, C, H, W)`
#     .. _DropBlock: A regularization method for convolutional networks:
#        https://arxiv.org/abs/1810.12890
#     """
#     def __init__(self, drop_ratio = 0.0, block_size = 7):
#         super(DropBlock2d, self).__init__()

#         self.drop_ratio  = drop_ratio
#         self.block_size = int(block_size)
#         self.eps        = 1e-8

#     def _compute_block_mask(self, mask):
#         block_mask = F.max_pool2d(input = mask[:, None, :, :], 
#                                 kernel_size = (self.block_size, self.block_size), 
#                                 stride = (1, 1), 
#                                 padding = self.block_size // 2)

#         if self.block_size % 2 == 0:
#             block_mask = block_mask[:, :, :-1, :-1]

#         return 1 - block_mask.squeeze(1)

#     def _compute_gamma(self, inputs):
#         return self.drop_ratio / (self.block_size ** 2)

#     def forward(self, inputs):

#         if self.drop_ratio == 0.0:
#             return inputs
#         else:
#             gamma = self._compute_gamma(inputs)
#             mask = (torch.rand(inputs.shape[0], * inputs.shape[2:]) < gamma).float()
#             mask = mask.to(inputs.device)
#             block_mask = self._compute_block_mask(mask)

#             out = inputs * block_mask[:, None, :, :]
#             return out * block_mask.numel() / block_mask.sum()


if __name__ == "__main__":
    from pprint import pprint
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    class tiny_model(nn.Module):

        def __init__ (self):
            super(tiny_model, self).__init__()
            self.layer1      = nn.Conv2d(3, 8, (1, 1))
            self.layer2      = nn.Conv2d(3, 8, (1, 1))
            self.layer3      = nn.Conv2d(3, 8, (1, 1))
        
        def forward (self, inputs):
            out = self.layer1(inputs)
            # out = self.layer2(inputs)
            # out = self.layer3(inputs)
            return out
        
    model  = tiny_model()
    model.to(device)
    inputs = torch.ones(1, 3, 3, 3)

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(lambda module, inp, out: sort(out))

    zerop = 5
    for i in range (20):
        if i % zerop == 0:
            drop = 0
            pass
        else:
            drop = np.random.uniform(0.0, 1.0)
        print(i)
        sort = SortDropout_v2(drop)
        inputs = inputs.to(device)
        out = model(inputs)
        print(out)

        # if i% zerop == 0:
        #     print(i)
        #     sort = SortDropout_v2(drop)
        #     inputs = inputs.to(device)
        #     out = model(inputs)
        #     print(out)