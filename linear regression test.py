import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data
true_w=torch.tensor([6,2,-2.6])
true_b=6.6
features,labels=d2l.synthetic_data(true_w,true_b,1000)
def iter(data_array,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
# 批量返回数据的迭代器
batch_size=10
data_iter=iter((features,labels),batch_size)
net=nn.Linear(3,1)
net.weight.data.normal_(0,0.01)
net.bias.data.fill_(0)
loss=nn.MSELoss(0)
trainer=torch.optim.SGD(net.parameters(),lr=0.05)
epochs=5
for epoch in range(epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    p=loss(net(features),labels)
    print(f'epoch {epoch+1},loss {p:f}')
print(net.weight.data,net.bias.data)
        