# Pruning Starter for Beginner
뉴럴 네트워크 프루닝 초보자를 위한 아주 간편한 튜토리얼  
- [Pytorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)  
- [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033)
- [An Overview of Neural Network Compression](https://arxiv.org/abs/2006.03669)  
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)  
- [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)
- [Structured DropConnect for Convolutional Neural Networks](http://www.cs.toronto.edu/~sajadn/sajad_norouzi/ECE1512.pdf)
  
뉴럴 네트워크 프루닝에는 몇 가지 관점이 있습니다.  
> Unstructured Pruning??? or structured Pruning???  
  
> One-shot Pruning??? or Iterative Pruning???  
  
1. Unsturctured Pruning  
  parameter 구조를 유지하면서, criteria에 따라 불필요한 weight는 0으로 만들어서 sparsity하게 만듭니다.  
2. Structured Pruning  
  parameter 구조 자체로 변형할 수 있습니다.  
  criteria에 따라 불필요한 weight는 0으로 만드는데, 직접 구조적인 맥락까지 바꿀 수 있는 방법입니다.  
3. One shot Pruning  
  네트워크를 한 번에 불필요한 weight를 빼서 프루닝하는 방법  
4. Iterative Pruning  
  네트워크를 조금씩 aggressive하게 프루닝하는 방법 (점진적으로)  
  
신경망에는 Dropout, DropBlock, DropBlock같은 다양한 regularization 방법이 있습니다.  
Regularization과 Pruning은 신경망이 Sparsity를 학습한다는 점에서 공통된 맥락이 있습니다.  
몇 가지 관련한 논문  
- [Learning Sparse Networks Using Targeted Dropout](https://arxiv.org/pdf/1905.13678.pdf)
- [Reducing Transformer Depth on Demand With Structured Dropout](https://arxiv.org/abs/1909.11556)
  
# Pruninf Prosses of Repository (for VGG model)
이 레포지토리에서는 VGG 모델만을 위한 훈련 및 저장 -> 로드 후 프루닝 -> 재훈련 및 저장의 과정을 제공합니다.  
  
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
