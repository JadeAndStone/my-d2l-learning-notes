import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
### original vgg_block
# def vgg_block(num_convs, in_channels, out_channels):
#     layers = []
#     for _ in range(num_convs):
#         layers.append(nn.Conv2d(in_channels, out_channels,
#                                 kernel_size=3, padding=1))
#         layers.append(nn.ReLU())
#         in_channels = out_channels
#     layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
#     return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
### new RepVGG
class RepVGG(nn.Module):
    def __init__(self,num_convs,in_channels,out_channels):
        super().__init__()
        self.g=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.vgg1=[]
        self.vgg1.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        self.vgg1.append(nn.BatchNorm2d(out_channels))
        self.vgg1.append(nn.ReLU())
        if num_convs>1:
            in_channels=out_channels
            self.vgg2=[]
            self.vgg2.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            self.vgg2.append(nn.BatchNorm2d(out_channels))
            self.vgg2.append(nn.ReLU())
            self.vgg2.append(nn.MaxPool2d(kernel_size=2,stride=2))
            self.vgg1=nn.Sequential(*self.vgg1,*self.vgg2)
        else:
            self.vgg1.append(nn.MaxPool2d(kernel_size=2,stride=2))
            self.vgg1=nn.Sequential(*self.vgg1)
        # self.bn1=nn.BatchNorm2d(out_channels)
        # self.bn2=nn.BatchNorm2d(out_channels)
        
    def forward(self,X):
        residual = self.g(X)
        Y=self.vgg1(X)
        Y+=self.g(X)
        if X.shape!=Y.shape:
            X=self.conv(X)
        Y+=X
        # print("Input shape:", X.shape)
        # print("Y shape:", Y.shape)
        
        # print("Residual branch output:", residual.shape)
        return F.relu(Y)
        
        
        
        
        
            
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(RepVGG(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())