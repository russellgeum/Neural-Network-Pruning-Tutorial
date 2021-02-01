# Pruning Starter for Beginner
프루닝 초보자를 위한 튜토리얼 안내서  
- [An Overview of Neural Network Compression](https://arxiv.org/abs/2006.03669)  
- [Pytorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)  
네트워크 프루닝에는 크게 두 가지 맥락이 있습니다.  
Unstructured Pruning, structured Pruning  
```
Unsturctured Pruning:
parameter 구조를 유지하면서, criteria에 따라 불필요한 weight는 0으로 만들어서 sparsity하게 만듬. 
Structured Pruning:
parameter 구조 자체로 변형할 수 있습니다. criteria에 따라 불필요한 weight는 0으로 만드는데,
직접 구조적인 맥락까지 변경할 수 있는 방법입니다.

Dropout, DropBlock, DropBlock같은 다양한 regularization 방법이 있습니다.
이 기법들은 training 시에 네트워크 구조에 sparsity한 성질을 부여하기 때문에
Iterative Pruning한 방법에 응용할 수 있습니다.

One shot Pruning:
네트워크를 한 번에 불필요한 weight를 빼서 프루닝하는 방법
Iterative Pruning:
네트워크를 조금씩 aggressive하게 프루닝하는 방법 (점진적으로)
```
# Summary of torch.prune module
## Define simply convolution module
```
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

net = nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size = (3, 3), stride = 1, padding = 1)

print(net.weight.shape)
==> torch.Size([2, 3, 3, 3])
```
## Exploring of CNN, DNN weights
```
모델 인스턴스의 내부 레이어에 접근해봅시다.  
module.weight에 접근하여 shape를 찍으면, [out_channel = 4, in_channel = 3, height = 3, width = 3]  
module.parameters()는 weight, bias에 해당하는 값만 출력  
module.named_paramters()는 각 파라미터와 weight, bias Key가 같이 출력  

print(list(module.parameters()))
[Parameter containing:
tensor([[[[ 0.1888, -0.0723, -0.0635],
          [-0.1858, -0.0566,  0.1365],
          [-0.0523, -0.0526,  0.1292]],

         [[-0.1476, -0.0387,  0.0431],
          [-0.1065,  0.0389,  0.0089],
          [ 0.0094, -0.0536, -0.1871]],

         [[-0.0716, -0.0330, -0.1025],
          [-0.1708, -0.1073,  0.0821],
          [ 0.1588,  0.0013, -0.0586]]],


        [[[ 0.0785, -0.1809,  0.0158],
          [-0.0923, -0.0121,  0.0697],
          [ 0.1339,  0.1373, -0.0528]],

         [[-0.1755,  0.1867,  0.0342],
          [ 0.1643, -0.0684, -0.0287],
          [-0.1600, -0.1110,  0.1027]],

         [[ 0.0119, -0.1053,  0.0603],
          [ 0.1612, -0.0757,  0.0062],
          [-0.1853,  0.1562,  0.1103]]]], requires_grad=True),
 Parameter containing:
tensor([ 0.0260, -0.1161], requires_grad=True)]

print(list(module.named_parameters()))
[('weight',
  Parameter containing:
tensor([[[[ 0.1888, -0.0723, -0.0635],
          [-0.1858, -0.0566,  0.1365],
          [-0.0523, -0.0526,  0.1292]],

         [[-0.1476, -0.0387,  0.0431],
          [-0.1065,  0.0389,  0.0089],
          [ 0.0094, -0.0536, -0.1871]],

         [[-0.0716, -0.0330, -0.1025],
          [-0.1708, -0.1073,  0.0821],
          [ 0.1588,  0.0013, -0.0586]]],


        [[[ 0.0785, -0.1809,  0.0158],
          [-0.0923, -0.0121,  0.0697],
          [ 0.1339,  0.1373, -0.0528]],

         [[-0.1755,  0.1867,  0.0342],
          [ 0.1643, -0.0684, -0.0287],
          [-0.1600, -0.1110,  0.1027]],

         [[ 0.0119, -0.1053,  0.0603],
          [ 0.1612, -0.0757,  0.0062],
          [-0.1853,  0.1562,  0.1103]]]], requires_grad=True)),
 ('bias', Parameter containing:
tensor([ 0.0260, -0.1161], requires_grad=True))]
```
## torch.prune.random_unstructured()
```
위의 모듈을 random_unstructured에 삽입하고,
어떤 파라미터를 프루닝할껀지? (name) 얼마나 프루닝할껀지? (amount) 설정 가능  
아래의 출력을 보면 직접 parameter에 접근하여 프루닝하지는 않으나,  
원래 parameter 정보가 orig가 붙은 키 값으로 변경   

prune.random_unstructured(module, name = "weight", amount = 0.5)
prune.random_unstructured(module, name = "bias", amount = 0.5)
print(list(module.named_parameters()))
[('weight_orig',
  Parameter containing:
tensor([[[[ 0.1888, -0.0723, -0.0635],
          [-0.1858, -0.0566,  0.1365],
          [-0.0523, -0.0526,  0.1292]],

         [[-0.1476, -0.0387,  0.0431],
          [-0.1065,  0.0389,  0.0089],
          [ 0.0094, -0.0536, -0.1871]],

         [[-0.0716, -0.0330, -0.1025],
          [-0.1708, -0.1073,  0.0821],
          [ 0.1588,  0.0013, -0.0586]]],


        [[[ 0.0785, -0.1809,  0.0158],
          [-0.0923, -0.0121,  0.0697],
          [ 0.1339,  0.1373, -0.0528]],

         [[-0.1755,  0.1867,  0.0342],
          [ 0.1643, -0.0684, -0.0287],
          [-0.1600, -0.1110,  0.1027]],

         [[ 0.0119, -0.1053,  0.0603],
          [ 0.1612, -0.0757,  0.0062],
          [-0.1853,  0.1562,  0.1103]]]], requires_grad=True)),
 ('bias_orig',
  Parameter containing:
tensor([ 0.0260, -0.1161], requires_grad=True))]
```
```
toch.prune 모듈은 parameter에 직접적 pruning하지 않는다.  
module.named_buffers()에 어떤 parameter를 pruning할지 binary 정보로 buffer에 저장  

print(list(module.named_buffers()))
[('weight_mask',
  tensor([[[[1., 1., 0.],
          [0., 1., 0.],
          [1., 1., 0.]],

         [[0., 0., 1.],
          [0., 0., 1.],
          [0., 0., 0.]],

         [[0., 1., 1.],
          [0., 1., 0.],
          [0., 0., 1.]]],


        [[[1., 0., 1.],
          [0., 1., 1.],
          [1., 0., 1.]],

         [[0., 0., 1.],
          [0., 1., 0.],
          [1., 1., 0.]],

         [[0., 1., 1.],
          [1., 1., 0.],
          [1., 0., 1.]]]])),
 ('bias_mask', tensor([1., 0.]))]
```
```
forward시에 작동하는 weight를 살펴봅시다.
print(moudle.weight)에는 module.paramters()에 module.buffers()가 적용되어
elementwise product 형태로 pruning한 param을 출력하는 것을 확인 가능

print(module.weight)
print(module.bias)
tensor([[[[ 0.1888, -0.0723, -0.0000],
          [-0.0000, -0.0566,  0.0000],
          [-0.0523, -0.0526,  0.0000]],

         [[-0.0000, -0.0000,  0.0431],
          [-0.0000,  0.0000,  0.0089],
          [ 0.0000, -0.0000, -0.0000]],

         [[-0.0000, -0.0330, -0.1025],
          [-0.0000, -0.1073,  0.0000],
          [ 0.0000,  0.0000, -0.0586]]],


        [[[ 0.0785, -0.0000,  0.0158],
          [-0.0000, -0.0121,  0.0697],
          [ 0.1339,  0.0000, -0.0528]],

         [[-0.0000,  0.0000,  0.0342],
          [ 0.0000, -0.0684, -0.0000],
          [-0.1600, -0.1110,  0.0000]],

         [[ 0.0000, -0.1053,  0.0603],
          [ 0.1612, -0.0757,  0.0000],
          [-0.1853,  0.0000,  0.1103]]]], grad_fn=<MulBackward0>)
tensor([0.0260, -0.0000], grad_fn=<MulBackward0>)
```
```
module._forward_pre_hooks을 통해서 레이어에 걸린 hooks가 prune 모듈임을 확인s
print(module._forward_pre_hooks)
OrderedDict([(86,
              <torch.nn.utils.prune.RandomUnstructured object at 0x7f2b661e72b0>),
             (87,
              <torch.nn.utils.prune.RandomUnstructured object at 0x7f2b661e7400>)])
```
## torch.prune.ln_structured()
```
prune.ln_structured()은 layer에 structured pruning을 실행
prune.random_unstructured() and prune.l1_unstructured()은 matrix를 sparsity하게 만드는 것
prune.ln_stuctured()은 통째로 filter를 0으로 만들어 버릴 수 있다. ==> 구조적으로 slim하게 가능

Ex 1)
module = nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size = (3, 3), stride = 1, padding = 1)
prune.ln_structured(module, name="weight", amount = 0.5, n = 2, dim = 0)

print(module.weight)
tensor([[[[-0.0000, -0.0000, -0.0000],
          [-0.0000,  0.0000,  0.0000],
          [-0.0000, -0.0000, -0.0000]],

         [[ 0.0000, -0.0000, -0.0000],
          [-0.0000, -0.0000, -0.0000],
          [-0.0000, -0.0000,  0.0000]],

         [[-0.0000, -0.0000, -0.0000],
          [-0.0000,  0.0000, -0.0000],
          [ 0.0000,  0.0000, -0.0000]]],


        [[[ 0.0594, -0.1061, -0.0484],
          [-0.1637,  0.1138, -0.1382],
          [-0.0441, -0.1791,  0.0292]],

         [[ 0.0411,  0.1839, -0.0214],
          [ 0.0745, -0.0854, -0.1346],
          [ 0.1871,  0.1144, -0.1441]],

         [[-0.1252, -0.1046, -0.0494],
          [ 0.1746,  0.0898,  0.1587],
          [ 0.0955,  0.0040,  0.1554]]]], grad_fn=<MulBackward0>)


Structured pruning을 더 작은 레이어에 이식하는 방법

Ex 2)
samll_module = nn.Conv2d(3, 1, (3, 3), 1, 1)
samll_module.weight.data = module.weight.data[1].clone()

print(samll_module.weight)
Parameter containing:
tensor([[[ 0.0594, -0.1061, -0.0484],
         [-0.1637,  0.1138, -0.1382],
         [-0.0441, -0.1791,  0.0292]],

        [[ 0.0411,  0.1839, -0.0214],
         [ 0.0745, -0.0854, -0.1346],
         [ 0.1871,  0.1144, -0.1441]],

        [[-0.1252, -0.1046, -0.0494],
         [ 0.1746,  0.0898,  0.1587],
         [ 0.0955,  0.0040,  0.1554]]], requires_grad=True)
```
# This repository...
## Directory
```
./folder
    /model
    /save
        /cifar10/vgg16_300.pt
        /cifqr100/vgg16_300.pt
    /prune
model_print.py
model_train.py
model_test.py
model_prune.py
module.py
```
## Usage
1. 모델을 최초로 한 번 학습하고 가중치 저장
```
python model_train.py --model vgg16 
                      --data cifar10 
                      --load none
                      --epoch 300
                      --batch 256
                      --optim sgd
                      --lr 0.1
                      --step 300
                      --size 32
                      --ver original (original 시에는 --ar args 무시)

학습이 다 끝나면 ./save/--data 폴더에 가중치 저장
```
2. 모델을 프루닝하고, 작은 모델에 이식하여 가중치 저장
```
python model_prune.py --model vgg16 --data cifar10 --load ./save/cifar10/vgg16_300.pt --ar 0.5 --po l1

프루닝이 다 끝나면 ./save/prune 폴더에 프루닝한 모델 가중치 저장
```
3. 프루닝한 모델을 로드하여 재학습
```
python model_train.py --model vgg16 
                      --data cifar10 
                      --load none
                      --epoch 300
                      --batch 256
                      --optim sgd
                      --lr 0.1
                      --step 300
                      --size 32
                      --ver pruned
                      --ar 0.5 (pruned 시에 작동, 2.에서 입력한 --ar 정보와 일치해야함)
```