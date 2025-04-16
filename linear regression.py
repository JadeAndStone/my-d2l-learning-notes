import torch
from d2l import torch as d2l
from torch.utils import data
from torch import nn
true_w=torch.tensor([1,2,-3.6])
true_b=6.6
features,labels=d2l.synthetic_data(true_w,true_b,1000)
def load_func(load_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*load_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
batch_size=10
data_iter=load_func((features,labels),batch_size)
loss=nn.MSELoss()
net=nn.Linear(3,1)
net.weight.data.normal_(0,0.01)
net.bias.data.fill_(0)
trainer=torch.optim.SGD(net.parameters(),lr=0.03)
epochs=3
for epoch in range(epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f'epoch {epoch+1},loss {l:f}')
print(net.weight.data,net.bias.data)