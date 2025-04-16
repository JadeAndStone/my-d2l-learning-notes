import torch
from torch import nn
def corr2d(X,K):
    r,c=K.shape
    Y=torch.zeros((X.shape[0]-r+1,X.shape[1]-c+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+r,j:j+c]*K).sum()
    return Y
# A=torch.tensor([[1,2,3],[1,0,1],[-1,0,2]])
# B=torch.tensor([[1,-1],[2,3]])
# print(corr2d(A,B))
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        self.weight=nn.Parameter(torch.rand(kernel_size))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,X):
        return corr2d(X,self.weight)+self.bias
X=torch.zeros(6,8)
X[:,2:6]=1
kernel=torch.tensor([[-1,1]])
Y=corr2d(X,kernel)
X=X.reshape((1,1,6,8))
Y=Y.reshape((1,1,6,7))
conv2d=nn.Conv2d(1,1,kernel_size=(2,2),bias=False)
# print(conv2d(X).shape[2:],X.shape[2:])
# lr=0.022
# for i in range(10):
#     Y_hat=conv2d(X)
#     l=(Y_hat-Y)**2
#     conv2d.zero_grad()
#     l.sum().backward()
#     conv2d.weight.data-=lr*conv2d.weight.grad
#     print(f'epoch:{i+1},loss:{l.sum():.3f}')
# print(conv2d.weight.data.reshape(1,2))
def corr2d_multi_in(X,K):
    return sum(conv2d(x,k) for x,k in zip(X,K))
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X,K))
