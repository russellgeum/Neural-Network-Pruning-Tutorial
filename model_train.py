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



class TRANING (object):
    def __init__ (self, args, device):
        self.args          = args
        self.device        = device
        self.best_accuracy = 0
        print("AVAILABLE DEVICE         :  ", device)
        print("SELECT MODE              :  ", self.args.model)
        print("SELECT DATASETS          :  ", self.args.data)
        # print("REGULARIZATION           :  ", self.args.opt)
        print("EPOCH, BATCH, STEP SIZE  :  ", self.args.epoch, self.args.batch, self.args.step)

        "DEFINE MODEL AND DATASETS"
        if self.args.ver == "original":
            if self.args.data == "cifar10":
                self.train_loader, self.test_loader = CIFAR10_datagenerator(batch_size = self.args.batch, size = self.args.size)
                self.model                          = call_model(model_name = self.args.model, num_classes = 10, load_path = self.args.load, device = self.device)
            elif self.args.data == "cifar100":
                self.train_loader, self.test_loader = CIFAR100_datagenerator(batch_size = self.args.batch, size = self.args.size)
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

        self.train_length = len(self.train_loader)
        self.test_length  = len(self.test_loader)
        summary(self.model.to(device), (3, self.args.size, self.args.size))

        "DEFINE OPTIMIZER AND LOSS"
        if self.args.optim   == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.lr, momentum = 0.9, weight_decay = 5e-4)
            print("OPTIMIZER   :   torch.optim.SGD")
        elif self.args.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr)
            print("OPTIMIZER   :   torch.optim.Adam")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.args.step, gamma = 0.1)


    def model_train (self, inputs, outputs, model):
        model.train()
        self.optimizer.zero_grad()
        
        pred       = model(inputs)
        loss       = self.criterion(pred, outputs)
        loss.backward()
        self.optimizer.step()
        is_correct = torch.argmax(pred, 1) == outputs
        batch_acc  = is_correct.float().mean()
        return loss, batch_acc


    def model_test (self, inputs, outputs, model):
        model.eval()

        pred       = model(inputs)
        loss       = self.criterion(pred, outputs)
        is_correct = torch.argmax(pred, 1) == outputs
        batch_acc  = is_correct.float().mean()
        return loss, batch_acc


    def loop(self):
        'Loop per epoch'
        for epo in tqdm(range (self.args.epoch)):
            train_losses   = 0
            train_accuracy = 0
            test_losses    = 0
            test_accuracy  = 0

            "Loop of train_loader"
            for inputs, outputs in self.train_loader:
                inputs, outputs   = inputs.to(self.device), outputs.to(self.device)
                tr1, ta1          = self.model_train(inputs, outputs, self.model)
                train_losses      = train_losses + tr1
                train_accuracy    = train_accuracy + ta1

            "Loop of test_loader"
            with torch.no_grad():
                for inputs, outputs in self.test_loader:
                    inputs, outputs  = inputs.to(self.device), outputs.to(device)
                    tr2, ta2         = self.model_test(inputs, outputs, self.model)

                    test_losses      = test_losses + tr2
                    test_accuracy    = test_accuracy + ta2

            train_losses    = train_losses / self.train_length
            train_accuracy  = train_accuracy  / self.train_length
            test_losses     = test_losses / self.test_length
            test_accuracy   = test_accuracy  / self.test_length

            print("--- EPOCH   ", epo + 1, " ---")
            print("TRAIN LOSS ", "{0:.4f}".format(train_losses.cpu().item()),         "        VALL LOSS   ",  "{0:.4f}".format(test_losses.cpu().item()))
            print("TRAIN ACC  ", "{0:.4f}".format(100*train_accuracy.cpu().item()), "%", "      VALL ACC    ",  "{0:.4f}".format(100*test_accuracy.cpu().item()), "%")
            # if test_accuracy > self.best_accuracy:
            #     torch.save(self.model.state_dict(), "./save/" + str(self.args.data) + "/" + "{:03d}".format(self.args.epoch) + ".pt")
            #     self.best_accuracy = test_accuracy
            self.scheduler.step()


    def model_save (self):
        if self.args.ver   == "original":
            torch.save(self.model.state_dict(), "./save/" + str(self.args.data) + "/" + "{:03d}".format(self.args.epoch) + ".pt")
        elif self.args.ver == "pruned":
            torch.save(self.model.state_dict(), "./save/prune/" + "retrained_" + str(self.args.model) + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train Option')
    parser.add_argument("--model", type = str,   default = "vgg16",     help = "deep learning model name")
    parser.add_argument("--data",  type = str,   default = "cifar10",   help = "CFIAR10 or CIFAR100")
    parser.add_argument("--load",  type = str,   default = "none",      help = "load file path")
    parser.add_argument("--epoch", type = int,   default = 300,         help = "training epoch")
    parser.add_argument("--batch", type = int,   default = 256,         help = "batch size")
    parser.add_argument("--optim", type = str,   default = "sgd",       help = "optimize")
    parser.add_argument("--lr",    type = float, default = 0.1,         help = "learning rate")
    parser.add_argument("--step",  type = int,   default = 100,         help = "learnig rate scheduler step")
    parser.add_argument("--size",  type = int,   default = 32,          help = "image size")
    parser.add_argument("--ver",   type = str,   default = "original",  help = "original or pruned")
    parser.add_argument("--ar",    type = float, default = 0.01,        help = "0.01 ~ 0.99")
    args = parser.parse_args()

    'SEED, MODEL.DEVICE(OPTION)'
    seed_everything()
    if not(os.path.isdir("./save")): os.makedirs("./save")

    TRAIN = TRANING(args, device)
    TRAIN.loop()
    TRAIN.model_save()