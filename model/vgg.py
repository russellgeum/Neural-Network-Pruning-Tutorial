import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


vgg_config = {'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
              'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
              'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
              'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def prune_cfg (cfg_list, alive_ratio):
    prune_cfg = {'prune' : []}
    for data in cfg_list:
        if type(data) is int:
            prune_cfg["prune"] = prune_cfg["prune"] + [int(alive_ratio * data)]
        else:
            prune_cfg["prune"] = prune_cfg["prune"] + [data]
            
    return prune_cfg



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
VGG MODEL CLASS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class VGG (nn.Module):

    def __init__(self, layer_list, num_classes):
        super(VGG, self).__init__()
        self.num_classes               = num_classes
        self.convolution, last_channel = layer_list
        self.classifier  = nn.Sequential(nn.Linear(last_channel, 512), 
                                        nn.ReLU(inplace = True), 
                                        nn.Linear(512, 512), 
                                        nn.ReLU(inplace = True), 
                                        nn.Linear(512, self.num_classes))

    def forward(self, inputs):
        outputs = self.convolution(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)

        return outputs



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
MAKE LAYERS MODULE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def make_layers (model_config_option, batch_norm = False):
    layers      = []
    in_channels = 3

    for config_option in model_config_option:
        if config_option == 'M':
            layers = layers + [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else: 
            conv2d = nn.Conv2d(in_channels, config_option, kernel_size = (3, 3), padding = 1)
            if batch_norm == False:
                layers = layers + [conv2d, nn.ReLU(inplace = True)]
            elif batch_norm == True:
                layers = layers + [conv2d, nn.BatchNorm2d(config_option), nn.ReLU(inplace = True)]

            in_channels = config_option
    return nn.Sequential(*layers), in_channels



def vgg11 (batch_norm = True, num_classes = 10):
    return VGG(layer_list = make_layers(vgg_config["vgg11"], batch_norm = batch_norm), num_classes = num_classes)


def vgg13 (batch_norm = True, num_classes = 10):
    return VGG(layer_list = make_layers(vgg_config["vgg13"], batch_norm = batch_norm), num_classes = num_classes)


def vgg16 (batch_norm = True, num_classes = 10):
    return VGG(layer_list = make_layers(vgg_config["vgg16"], batch_norm = batch_norm), num_classes = num_classes)


def vgg19 (batch_norm = True, num_classes = 10):
    return VGG(layer_list = make_layers(vgg_config["vgg19"], batch_norm = batch_norm), num_classes = num_classes)


def slim_vgg (vgg_name = "vgg16", alive_ratio = 0.5, batch_norm = True, num_classes = 10):
    config = prune_cfg(vgg_config[vgg_name], alive_ratio = alive_ratio)
    # print(config)
    return VGG(layer_list = make_layers(config['prune'], batch_norm = batch_norm), num_classes = num_classes)