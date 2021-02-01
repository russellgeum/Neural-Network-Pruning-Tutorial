# -*- conding: utf-8 -*-
import os
import sys
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from module import *
from model.vgg import *
from model.resnet import *
from model.mobilenetv2 import *

from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# def montecarlo_pruning(new_model, old_model):
#     LIMIT_STATIC_NUMBER = 200
#     LOWER_BOUND_RATIO   = 0.97
#     layer_count_number  = 0
#     static_count_number = 0
#     prev_index = channel_random_pruning(old_model, 1 - alive_ratio)

#     for count in range (100000):
#         # print(count, layer_count_number)
#         print(layer_count_number)
#         '1. Prev accuracy new_model'
#         old_model     = call_model(args.model, args.opt, 0.0, 1.0, args.load)
#         oldmodel2newmodel(new_model.to(device), old_model.to(device), prev_index)
#         prev_result   = inference(new_model.to(device), test_loader)
#         remeber_model = copy.deepcopy(new_model)

#         '2. Get preserved current channel_index (From the front layer) //  After accuracy new_model'
#         old_model                       = call_model(args.model, args.opt, 0.0, 1.0, args.load)
#         next_index                      = channel_random_pruning(old_model, 1 - alive_ratio)
#         next_index[:layer_count_number] = prev_index[:layer_count_number] # Implantation current_index from prev_index[count] to weight_index2[count]
#         oldmodel2newmodel(new_model.to(device), old_model.to(device), next_index)
#         next_result                     = inference(new_model.to(device), test_loader)

#         '''
#         3. Print prev accruacy and next accuracy
#         if next accuracy higher than prev accuracy, 
#         layer index state of prev accuracy model would be replaced to layer index state of next accuracy
#         '''
#         print_accuracy(prev_result)
#         print_accuracy(next_result)

#         if prev_result.item() < next_result.item():
#             prev_index = next_index
#             next_index = []
#             layer_count_number = layer_count_number + 1
#             static_count_number = 0
#         elif LOWER_BOUND_RATIO * prev_result.item() < next_result.item() and next_result.item() <= prev_result.item():
#             prev_index = next_index
#             next_index = []
#             static_count_number = 0
#             pass
#         elif next_result.item() < prev_result.item():
#             static_count_number = static_count_number + 1
#             if LIMIT_STATIC_NUMBER < static_count_number:
#                 if LOWER_BOUND_RATIO * LOWER_BOUND_RATIO * prev_result.item() < next_result.item():
#                     prev_index = next_index
#                     next_index = []
#                     layer_count_number = layer_count_number + 1
#                     static_count_number = 0
#                 else:
#                     static_count_number = static_count_number + 1
#             pass 

#         if layer_count_number == 13:
#             print("EXIT MONTE CARLO SEARCH FOR GOOD PRUNING ARCHITECTURE")
#             new_model = copy.deepcopy(remeber_model)
#             return new_model
#             break

#     new_model = copy.deepcopy(remeber_model)
#     return new_model


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Mapping fuction for VGG13, 16, scale
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def alive_ratio_mapping (alive_ratio):
    for channels in range (64):
        if alive_ratio == 0.0:
            raise "args.pr is 0.0, Dont put 0.0 value to args.pr"
        if 0.015625 * channels > alive_ratio:
            map_alive_ratio = 0.015625 * (channels-1)
            return map_alive_ratio


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Calculate alinved channel index
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def alive_channel_index (module):
    mask        = module.cpu().weight_mask.numpy()
    mask        = np.mean(mask, axis = -1)
    mask        = np.mean(mask, axis = -1)
    mask        = np.mean(mask, axis = -1)
    alive_index = np.argwhere(mask)
    alive_index = np.squeeze(np.asarray(alive_index))
    alive_index = alive_index.tolist()
    return alive_index


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Structured pruning module for model that no have residual connection 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def channel_pruning (old_model, pruning_option = "l1", name = "weight", prune_ratio = 0.5):
    alive_weight_index = []
    
    # Get alived channel index via L1 Structured Pruning
    if pruning_option == "l1":
        for _, old_module in old_model.named_modules():
            if isinstance (old_module, torch.nn.Conv2d):
                prune.ln_structured(old_module, name = name, amount = prune_ratio, n = 1, dim = 0)
                alive_index = alive_channel_index(old_module)
                alive_weight_index.append(alive_index)

    # Get alived channel index via L2 Structured Pruning
    elif pruning_option == "l2":
        for _, old_module in old_model.named_modules():
            if isinstance (old_module, torch.nn.Conv2d):
                prune.ln_structured(old_module, name = name, amount = prune_ratio, n = 2, dim = 0)
                alive_index = alive_channel_index(old_module)
                alive_weight_index.append(alive_index)
    
    # Get alived cannel index via random Structured Pruning
    elif pruning_option == "random":
        for _, old_module in old_model.named_modules():
            if isinstance (old_module, nn.Conv2d):
                num_out_channel    = old_module.weight.data.shape[0]
                num_pruned_channel = int(prune_ratio * num_out_channel)
                num_alive_channel  = num_out_channel - num_pruned_channel
                alive_index        = random.sample(range(num_out_channel), num_alive_channel)
                alive_weight_index.append(alive_index)

    return alive_weight_index


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Deepcopy weight of model (from old_modl to new_model)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def oldmodel2newmodel (new_model, old_model, alive_weight_index):
    inchannel_index          = [0, 1, 2]
    convolution_module_count = 0
    outchannel_index         = alive_weight_index[convolution_module_count]

    for [new_module, old_module] in zip (new_model.modules(), old_model.modules()):
        if isinstance (old_module, nn.Conv2d):
            convolution_weight       = old_module.weight.data[ :, inchannel_index, :, :].clone()
            convolution_weight       = convolution_weight[outchannel_index, :, :, :].clone()
            new_module.weight.data   = convolution_weight.clone()
        
        elif isinstance (old_module, nn.BatchNorm2d):
            new_module.weight.data   = old_module.weight.data[outchannel_index].clone()
            old_module.bias.data     = old_module.bias.data[outchannel_index].clone()
            new_module.running_mean  = old_module.running_mean[outchannel_index].clone()
            new_module.running_var   = old_module.running_var[outchannel_index].clone()

            # conv -> batchnorm -> (coutning module) -> cov -> batchnorm
            convolution_module_count = convolution_module_count + 1
            inchannel_index          = outchannel_index
            if convolution_module_count < len(alive_weight_index):
                outchannel_index     = alive_weight_index[convolution_module_count]

        elif isinstance (old_module, nn.Linear):
            if convolution_module_count == len(alive_weight_index):
                weight_matrix_index      = alive_weight_index[-1]
                new_module.weight.data   = old_module.weight.data[ :, weight_matrix_index].clone()
                convolution_module_count = convolution_module_count + 1
                continue
            new_module.weight.data   = old_module.weight.data.clone()
            new_module.bias.data     = old_module.bias.data.clone()

        elif isinstance(old_module, nn.BatchNorm1d):
            new_module.weight.data   = old_module.weight.data.clone()
            new_module.bias.data     = old_module.bias.data.clone()
            new_module.running_mean  = old_module.running_mean.clone()
            new_module.running_var   = old_module.running_var.clone()


if __name__ == "__main__":
    from pprint import pprint
    parser = argparse.ArgumentParser(description = 'TEST')
    parser.add_argument("--model",type = str,   default = "vgg16",   help = "vgg11, vgg13, vgg16, vgg19")
    parser.add_argument("--data", type = str,   default = "cifar10", help = 'cfiar10 cifar100')
    parser.add_argument("--load", type = str,   default = "none",    help = "load file path")
    parser.add_argument("--ar",   type = float, default = 0.0,       help = "prune ratio")
    parser.add_argument("--po",   type = str,   default = "l1",      help = "prune option")
    args = parser.parse_args()

    alive_ratio = alive_ratio_mapping(args.ar)
    print(alive_ratio)

    print("INITIAL ACCURACY for VGG NETWORK")
    if args.data   == "cifar10":
        train_loader, test_loader = CIFAR10_datagenerator(128, 32)
        old_model = call_model(args.model, 10, args.load, device)
    elif args.data == 'cifar100':
        train_loader, test_loader = CIFAR100_datagenerator(128, 32)
        old_model = call_model(args.model, 100, args.load, device)

    old_model.to(device)
    summary(old_model, (3, 32, 32))
    accuracy            = inference(old_model, test_loader)
    print_accuracy(accuracy)

    print("CALL NEW MODEL and IMPLAT PARAMETERS")
    if args.data   == "cifar10":
        new_model = slim_vgg(vgg_name = args.model, alive_ratio = alive_ratio, num_classes = 10)
    elif args.data == 'cifar100':
        new_model = slim_vgg(vgg_name = args.model, alive_ratio = alive_ratio, num_classes = 100)

    alive_channel_index = channel_pruning(old_model, pruning_option = args.po, prune_ratio = 1-alive_ratio)
    oldmodel2newmodel(new_model, old_model, alive_channel_index)

    new_model.to(device)
    summary(new_model, (3, 32, 32))
    accuracy            = inference(new_model, test_loader)
    print_accuracy(accuracy)

    print("SAVE PRUNED MODEL")
    torch.save(new_model.state_dict(), "./save/prune/" + "pruned_" + str(args.model) + ".pt")