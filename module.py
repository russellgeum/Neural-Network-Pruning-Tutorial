# -*- conding: utf-8 -*-
import os
import sys
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
from natsort import natsorted
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

import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets import ImageFolder

from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


'SEED EVERYTHING'
def seed_everything (seed = 1):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


'CIFAR_DATAGENERATOR'
def CIFAR10_datagenerator (batch_size, size):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                                    
    cifar10_train = CIFAR10(root = './data', train = True, transform = transform_train, download = True)
    cifar10_test  = CIFAR10(root = './data', train = False, transform = transform_test, download = True)
    train_loader  = DataLoader(dataset = cifar10_train, batch_size = batch_size, shuffle = True, drop_last = False)
    test_loader   = DataLoader(dataset = cifar10_test, batch_size = batch_size, shuffle = False, drop_last = False)
    
    return train_loader, test_loader


'CIFAR100_DATAGENERATOR'
def CIFAR100_datagenerator (batch_size, size):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test  = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                                    
    cifar10_train = CIFAR100(root = './data', train = True, transform = transform_train, download = True)
    cifar10_test  = CIFAR100(root = './data', train = False, transform = transform_test, download = True)
    train_loader  = DataLoader(dataset = cifar10_train, batch_size = batch_size, shuffle = True, drop_last = False)
    test_loader   = DataLoader(dataset = cifar10_test, batch_size = batch_size, shuffle = False, drop_last = False)
    
    return train_loader, test_loader


'IMAGENET DATASETS'
class IMAGENET_datasets:
    '''
    Requirements
    train_image_path : train image path of imagenet
    valid_image_path : valid image path of imagenet
    valid_class_file : infor.csv located in val folder, So call it
    이미지넷의 train 이미지, valid 이미지의 파일들을 모두 불러온다.
    valid 이미지의 클래스 정보는 info.csv 형태로 되어있다. 그 중 "label" 부분을 따와서 train_datasets 처럼 구성해준다.
    '''
    # def get_datafram (class_info, category = "label"):
    #     category_list  = class_info["label"]
    #     category_list  = category_list.tolist()

    #     return category_list
        

    # def get_onehotencoding (class_list):
    #     encoder = LabelEncoder()
    #     encoder.fit(class_list)
    #     labels = encoder.transform(class_list)
    #     # labels = labels.reshape(-1, 1)

    #     # one_hot_encoder = OneHotEncoder()
    #     # one_hot_encoder.fit(labels)
    #     # one_hot_labels = one_hot_encoder.transform(labels)
    #     # one_hot_labels = one_hot_labels.toarray()

    #     return labels
    def __init__ (self,     
                train_image_path = "./imagenet/train",
                valid_image_path = "./imagenet/val/images", 
                size = 32):
        print("IMAGENET DATASLOADER TEST")
        self.train_image_path = train_image_path
        self.valid_image_path = valid_image_path
        self.size             = size
        # self.valid_class_file = pd.read_csv(valid_image_path + "/info.csv")
        # self.class_list  = get_datafram(self.valid_class_file, "label")
        # self.test_label = get_onehotencoding(self.class_list)
        # self.test_label = torch.Tensor(self.test_label)

    def create_train_datasets (self):
        '''
        이미지넷의 훈련 이미지는 클래스 별로 이미지가 들어가 있어서 ImageFolder가 알아서 labeling을 해준다.
        따라서 labeling 문제는 걱정하지 말고 그대로 쓰자

        for X, Y in self.train_datasets:
            print(X, Y)
            >>> (이미지 매트릭스, 레이블링 숫자)
        '''
        train_datasets = ImageFolder(self.train_image_path, 
                                    transforms.Compose([transforms.Resize((self.size, self.size)),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        return train_datasets
    
    def create_test_datasets (self):
        '''
        이미지넷의 발리드 이미지는 모두 한 폴더에 섞여 있다. 따라서 ImageFolder는 이를 알아서 labeling하지 못한다.
        infor.csv 파일에 보면 "label" 정보에 이미지의 라벨링이 있다. 이를 가져와서 self.test_datasets과 묶어주어야 한다.

        for (X, _), label in zip(self.test_datasets, self.category_label)
        '''
        test_datasets = ImageFolder(self.valid_image_path, 
                                    transforms.Compose([transforms.Resize((self.size, self.size)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        return test_datasets

    def return_datasets (self):
        self.train_datasets = self.create_train_datasets()
        self.test_datasets  = self.create_test_datasets()
        return self.train_datasets, self.test_datasets


'IMAGENET DATAGENERATOR'
def ImageNet_datagenerator(batch_size, size):
    datasets = IMAGENET_datasets(size = size)
    train_datasets, test_datasets = datasets.return_datasets()

    train_loader = DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True, drop_last = False)
    test_loader  = DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False, drop_last = False)

    return train_loader, test_loader


'MODEL BUILD'
def call_model(model_name, num_classes, load_path, device):      
    if model_name   == "vgg11":
        model = vgg11(True, num_classes = num_classes)
    elif model_name == "vgg13":
        model = vgg13(True, num_classes = num_classes)
    elif model_name == "vgg16":
        model = vgg16(True, num_classes = num_classes)
    elif model_name == "vgg19":
        model = vgg19(True, num_classes = num_classes)
    elif model_name == "resnet50":
        model = ResNet50(in_channels = 64, num_classes = num_classes)
    elif model_name == "resnet34":
        model = ResNet34(in_channels = 64, num_classes = num_classes)
    elif model_name == "mobilenetv2":
        # model = mobilenetv2(num_classes = num_classes)
        model = MobileNetV2(num_classes = num_classes)

    if load_path == "none":
        print("NO EXIST LOAD PATH...")
        pass
    else:
        model.load_state_dict(torch.load(os.path.join(load_path), map_location = device), strict = False)

    return model

'MODEL INFERENCE'
def inference (model, data_loader):
    accuracy = 0
    model.eval()

    with torch.no_grad ():
        for X, Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)
            prediction = model(X)
            correct_prediction  = (torch.argmax(prediction, 1) == Y)
            test_batch_accuracy = correct_prediction.float().mean()
            accuracy += test_batch_accuracy
        accuracy = accuracy / len(data_loader)

    return accuracy


'PRINT ACCURACY'
def print_accuracy (accuracy):
    print("ACCURACY  :                  {0:.2f}".format(100*accuracy))