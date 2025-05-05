import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
from torch.utils import data
import torchvision
from torchvision import transforms,datasets
class Residual(nn.Module):
    def __init__(self,input_channels,output_channels, use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(output_channels,output_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,output_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(output_channels)
        self.bn2=nn.BatchNorm2d(output_channels)
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)
b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
def resnet_block(input_channels,output_channels,num_residual,first_block=False):
    blk=[]
    for i in range(num_residual):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,output_channels,use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(output_channels,output_channels))
    return blk
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
# X=torch.rand((1,1,224,224))
# for layer in net:
#     X=layer(X)
#     print(layer.__class__.__name__,X.shape)
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
def Res_sgd():
    
    return
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
data_form=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
trainset=datasets.MNIST(root='./fashion_mnist',transform=data_form,download=True)
testset=datasets.MNIST(root='./fashion_mnist',transform=data_form,download=True)
