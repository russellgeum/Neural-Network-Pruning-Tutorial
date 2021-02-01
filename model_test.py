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


class INFERENCE:
      def __init__ (self, args, device):
            self.args   = args
            self.device = device
            print("INITIAL ACCURACY FOR NETWORK")
            print("MODEL   :", self.args.model)
            print("DATA    :", self.args.data)
            print("LOAD    :", self.args.load)

            "DEFINE MODEL AND DATASETS"
            if self.args.ver == "original":
                  if self.args.data == "cifar10":
                        self.train_loader, self.test_loader = CIFAR10_datagenerator(batch_size = 128, size = 32)
                        self.model                          = call_model(model_name = self.args.model, num_classes = 10, load_path = self.args.load, device = self.device)
                  elif self.args.data == "cifar100":
                        self.train_loader, self.test_loader = CIFAR100_datagenerator(batch_size = 128, size = 32)
                        self.model                          = call_model(model_name = self.args.model, num_classes = 100, load_path = self.args.load, device = self.device)
                  # elif self.args.data == "imagenet":
                  #     self.train_loader, self.test_loader = ImageNet_datagenerator(batch_size = self.args.batch, size = self.args.size)
                  #     self.model                          = call_model(model_name = self.args.model, num_classes = 1000, load_path = self.args.load, device = self.device)
            elif self.args.ver == "pruned":
                  if args.data   == "cifar10":
                        self.train_loader, self.test_loader = CIFAR10_datagenerator(batch_size = self.args.batch, size = self.args.size)
                        self.model                          = slim_vgg(vgg_name = self.args.model, alive_ratio = self.args.ar, num_classes = 10)
                  elif args.data == 'cifar100':
                        self.train_loader, self.test_loader = CIFAR100_datagenerator(batch_size = self.args.batch, size = self.args.size)
                        self.model                          = slim_vgg(vgg_name = self.args.model, alive_ratio = self.args.ar, num_classes = 100)

      def test (self):
            accuracy = inference(self.model.to(self.device), self.test_loader)
            print_accuracy(accuracy)


if __name__ == "__main__":
      parser = argparse.ArgumentParser(description = 'TEST')
      parser.add_argument("--model", type = str,   default = "vgg16",     help = "vgg11, vgg13, vgg16, vgg19, resnet50, mobilenetv2")
      parser.add_argument("--data",  type = str,   default = "cifar10",   help = "cifar10, cifar100, imagenet")
      parser.add_argument("--load",  type = str,   default = "default",   help = "load file path")
      parser.add_argument("--ver",   type = str,   default = "original",  help = "original or pruned")
      parser.add_argument("--ar",    type = float, default = 0.5,         help = "0.01 ~ 0.99")
      args = parser.parse_args()
      exe  = INFERENCE(args, device)
      exe.test()