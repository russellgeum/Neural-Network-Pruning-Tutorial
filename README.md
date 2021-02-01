# Pruning Starter for Beginner
프루닝 초보자를 위한 튜토리얼 안내서
- [An Overview of Neural Network Compression](https://arxiv.org/abs/2006.03669)
- [Pytorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
# Background
네트워크 프루닝에는 크게 두 가지 맥락이 있다. Unstructured Pruning, structured Pru
# Summary of torch.prune module
## Define simply module
아래와 같은 간단한 네트워크 모듈을 생각해봅시다.
입력을 6x6 사이즈의 3채널 이미지를 입력 받는다고 정의 해볼게요.
따라서 컨볼루션의 필터 1개는 3x3 커널의 3채널이고, 이 필터의 갯수를 4로 설정하였습니다.
```
class NetworkModule(nn.Module):
    def __init__(self):
        super(NetworkModule, self).__init__()
        self.conv = nn.Conv2d(in_channels = 3,out_channels = 4, kernel_size = (3, 3), stride = 1, padding = 1)
        self.fc   = nn.Linear(in_features = 4 * 3 * 3, out_features = 10) 

    def forward(self, input):
        output = self.conv(input)
        output = F.relu(output)
        output = F.max_pool2d(output, (2, 2))
        b, c, h, w = output.shape

        output = output.view(-1, c*h*w)
        output = self.fc(output)
        output = F.relu(output)

        return output

net = NetworkModule()

from torchsummary import summary
summary(net, (3, 6, 6))

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1              [-1, 4, 6, 6]             112
            Linear-2                   [-1, 10]             370
================================================================
Total params: 482
Trainable params: 482
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
```
