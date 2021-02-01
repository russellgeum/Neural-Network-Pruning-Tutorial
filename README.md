# Pruning Starter for Beginner
프루닝 초보자를 위한 튜토리얼 안내서
- [An Overview of Neural Network Compression](https://arxiv.org/abs/2006.03669)
- [Pytorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
# Background
네트워크 프루닝에는 크게 두 가지 맥락이 있습니다.  
Unstructured Pruning, structured Pruing  
# Summary of torch.prune module
## Define simply convolution module
```
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size = (3, 3), stride = 1, padding = 1)
print(net.weight.shape)

==> torch.Size([2, 3, 3, 3])
```
## Exploring of CNN, DNN weights
모델 인스턴스의 내부 레이어에 접근해봅시다.  
module.weight에 접근하여 shape를 찍으면 [out_channel = 4, in_channel = 3, height = 3, width = 3]  
module.parameters()는 weight, bias에 해당하는 값만 출력  
module.named_paramters()는 각 파라미터와 weight, bias Key가 같이 출력  
```
module.paramters()에 접근하면 requires_grad = True인 weight와 bias가 출력
print(list(module.parameters()))

==>
[Parameter containing:
tensor([[[[ 0.0589,  0.1299, -0.0005],
          [-0.1727,  0.1506,  0.0735],
          [-0.0755,  0.0601,  0.0035]],

         [[ 0.0165,  0.0076, -0.1838],
          [-0.1409, -0.1897, -0.0630],
          [-0.0387,  0.0753, -0.0818]],

         [[-0.0442, -0.1158, -0.1221],
          [-0.1007, -0.1160, -0.1600],
          [ 0.0301,  0.1706,  0.1659]]],


        [[[-0.1792,  0.1193,  0.0095],
          [-0.0481,  0.0839,  0.0946],
          [-0.1069, -0.0513, -0.1515]],

         [[-0.0372, -0.1030, -0.1872],
          [-0.1359, -0.1836,  0.1591],
          [-0.1401,  0.1586, -0.0205]],

         [[ 0.1408, -0.1248, -0.0774],
          [-0.0020, -0.0417, -0.0028],
          [-0.0520,  0.1164, -0.0901]]]], requires_grad=True), 
Parameter containing:
tensor([ 0.1234, -0.1730], requires_grad=True)]

module.named_parameters()에 접근하면 requires_grad = True인  
weight와 bias가 각각에 해당하는 "weight", "bias"가 키 값으로 함께 출력  
print(list(module.named_parameters()))

==>
[('weight', Parameter containing:
tensor([[[[ 0.0589,  0.1299, -0.0005],
          [-0.1727,  0.1506,  0.0735],
          [-0.0755,  0.0601,  0.0035]],

         [[ 0.0165,  0.0076, -0.1838],
          [-0.1409, -0.1897, -0.0630],
          [-0.0387,  0.0753, -0.0818]],

         [[-0.0442, -0.1158, -0.1221],
          [-0.1007, -0.1160, -0.1600],
          [ 0.0301,  0.1706,  0.1659]]],


        [[[-0.1792,  0.1193,  0.0095],
          [-0.0481,  0.0839,  0.0946],
          [-0.1069, -0.0513, -0.1515]],

         [[-0.0372, -0.1030, -0.1872],
          [-0.1359, -0.1836,  0.1591],
          [-0.1401,  0.1586, -0.0205]],

         [[ 0.1408, -0.1248, -0.0774],
          [-0.0020, -0.0417, -0.0028],
          [-0.0520,  0.1164, -0.0901]]]], requires_grad=True)),
('bias', Parameter containing:
tensor([ 0.1234, -0.1730], requires_grad=True))]
```
# torch.prune.random_unstructured()
위의 모듈을 random_unstructured에 삽입하고  
어떤 파라미터를 프루닝할껀지? (name) 얼마나 프루닝할껀지? (amount) 설정 가능  
아래의 출력을 보면 직접 parameter에 접근하여 프루닝하지는 않으나,  
원래 parameter 정보가 orig가 붙은 키 값으로 변경   
```
prune.random_unstructured(module, name = "weight", amount = 0.5)
prune.random_unstructured(module, name = "bias", amount = 0.5)
print(list(module.named_parameters()))

==>
[('weight_orig', Parameter containing:
tensor([[[[ 0.0589,  0.1299, -0.0005],
          [-0.1727,  0.1506,  0.0735],
          [-0.0755,  0.0601,  0.0035]],

         [[ 0.0165,  0.0076, -0.1838],
          [-0.1409, -0.1897, -0.0630],
          [-0.0387,  0.0753, -0.0818]],

         [[-0.0442, -0.1158, -0.1221],
          [-0.1007, -0.1160, -0.1600],
          [ 0.0301,  0.1706,  0.1659]]],


        [[[-0.1792,  0.1193,  0.0095],
          [-0.0481,  0.0839,  0.0946],
          [-0.1069, -0.0513, -0.1515]],

         [[-0.0372, -0.1030, -0.1872],
          [-0.1359, -0.1836,  0.1591],
          [-0.1401,  0.1586, -0.0205]],

         [[ 0.1408, -0.1248, -0.0774],
          [-0.0020, -0.0417, -0.0028],
          [-0.0520,  0.1164, -0.0901]]]], requires_grad=True)),
('bias_orig', Parameter containing:
tensor([ 0.1234, -0.1730], requires_grad=True))]
```
---
toch.prune 모듈은 parameter에 직접적으로 pruning하지 않는다.  
module.named_buffers()에 어떤 parameter를 pruning할지 binary mask 정보로 버퍼에 저장  
```
print(list(module.named_buffers()))

==>
[('weight_mask', 
tensor([[[[0., 0., 1.],
          [1., 1., 1.],
          [0., 1., 1.]],

         [[0., 1., 0.],
          [0., 0., 0.],
          [1., 0., 0.]],

         [[0., 1., 1.],
          [1., 1., 1.],
          [1., 0., 1.]]],


        [[[0., 1., 0.],
          [1., 1., 0.],
          [1., 1., 0.]],

         [[1., 1., 1.],
          [0., 0., 0.],
          [1., 1., 0.]],

         [[0., 1., 0.],
          [0., 0., 0.],
          [0., 0., 1.]]]])), 
('bias_mask', 
tensor([0., 1.]))]
```
---
forward시에 작동하는 weight를 살펴봅시다.
print(moudle.weight)에는 module.paramters()에 module.buffers()가 적용되어 (elementwise product)
프루닝돈 것을 확인 가능
```
print(module.weight)
print(module.bias)

==>
tensor([[[[ 0.0000,  0.0000, -0.0005],
          [-0.1727,  0.1506,  0.0735],
          [-0.0000,  0.0601,  0.0035]],

         [[ 0.0000,  0.0076, -0.0000],
          [-0.0000, -0.0000, -0.0000],
          [-0.0387,  0.0000, -0.0000]],

         [[-0.0000, -0.1158, -0.1221],
          [-0.1007, -0.1160, -0.1600],
          [ 0.0301,  0.0000,  0.1659]]],


        [[[-0.0000,  0.1193,  0.0000],
          [-0.0481,  0.0839,  0.0000],
          [-0.1069, -0.0513, -0.0000]],

         [[-0.0372, -0.1030, -0.1872],
          [-0.0000, -0.0000,  0.0000],
          [-0.1401,  0.1586, -0.0000]],

         [[ 0.0000, -0.1248, -0.0000],
          [-0.0000, -0.0000, -0.0000],
          [-0.0000,  0.0000, -0.0901]]]], grad_fn=<MulBackward0>)
tensor([ 0.0000, -0.1730], grad_fn=<MulBackward0>)
```
---
print(module._forward_pre_hooks)을 통해서 레이어에 걸린 hooks가 prune 모듈임을 확인
```
print(module._forward_pre_hooks)
==> OrderedDict([(68, <torch.nn.utils.prune.RandomUnstructured object at 0x7f2b66321278>) ~~
```
