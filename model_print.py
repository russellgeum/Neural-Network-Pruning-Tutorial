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
from model.densenet import *

from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train Option')
    parser.add_argument("--model", type = str,  default = "vgg16",  help = "deep learning model name")
    parser.add_argument("--num",   type = int,  default = 10,       help = "number of classes")
    parser.add_argument("--size",  type = int,  default = 32,       help = "image size")
    args = parser.parse_args()

    if args.model   == "vgg11":
        model = vgg11(True, num_classes = args.num)
    elif args.model == "vgg13":
        model = vgg13(True, num_classes = args.num)
    elif args.model == "vgg16":
        model = vgg16(True, num_classes = args.num)
    elif args.model == "vgg19":
        model = vgg19(True, num_classes = args.num)
    elif args.model == "resnet50":
        model = ResNet50(in_channels = 64, num_classes = args.num)
    elif args.model == "resnet34":
        model = ResNet34(in_channels = 64, num_classes = args.num)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(num_classes = args.num)

    model.to(device)
    summary(model, (3, args.size, args.size))