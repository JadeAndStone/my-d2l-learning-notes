import torch
import numpy as np
from torch.utils import data
from torchvision import datasets,transforms
from multiprocessing import freeze_support
import math
from torch import nn
from sklearn.model_selection import KFold
def main():
    max_degree=50
    train_size=500
    test_size=train_size/5
    true_w=torch.zeros(max_degree)
    true_w[0:4]=torch.tensor([5, 1.2,-3.4,2.2])
    feature=torch.randn(train_size)
    labels=torch.zeros(train_size)
    def func(x):
        temp=(torch.tensor(np.power(x.numpy(),np.arange(max_degree)))).type(true_w.dtype)
        for i in range(max_degree):
            temp[i]/=math.gamma(i+1)
        return temp
    for i in range(train_size):
        labels[i]=torch.dot((torch.tensor(np.power((feature[i]).numpy(),np.arange(max_degree)))).type(true_w.dtype),true_w)
    def iter(data_array,batch_size,is_train=True):
        dataset=data.TensorDataset(*data_array)
        return data.DataLoader(dataset,batch_size,shuffle=is_train)
    features=torch.zeros(train_size,max_degree)    
    for i in range(train_size):
        features[i]=func(feature[i])
    net=nn.Sequential(nn.Linear(max_degree,max_degree*10),nn.ReLU(),nn.Dropout(0.1),
                      nn.Linear(max_degree*10,max_degree*5),nn.ReLU(),nn.Dropout(0.1),
                      nn.Linear(max_degree*5,1))
    loss=nn.MSELoss(reduction='mean')
    trainer=torch.optim.SGD(net.parameters(),lr=0.01)
    num_epoch=10
    KF=KFold(n_splits=5)
    t=0
    lamda=0.1
    for epoch in range(num_epoch):
        n_loss,train_loss=0,0
        for train_index,test_index in KF.split(features):
            t+=1
            if t==2:
                break
            features_train,features_test=features[train_index],features[test_index]
            labels_train,labels_test=labels[train_index],labels[test_index]
            data_iter=iter((features_train,labels_train),100)
            test_iter=iter((features_test,labels_test),100)
            net.train()
            for X,y in data_iter:
                trainer.zero_grad()
                l=loss(net(X),y.reshape(net(X).shape))+lamda/2*(torch.norm(net[0].weight.data))**2+lamda/2*(torch.norm(net[3].weight.data))**2
                l.backward()
                trainer.step()
                train_loss+=loss(net(X),y.reshape(net(X).shape))
            net.eval()
            with torch.no_grad():
                for X,y in test_iter:
                    n_loss+=loss(net(X),y.reshape(net(X).shape))
        print(f'epoch{epoch+1},train_loss:{train_loss/train_size:f},test_loss={n_loss/test_size:f}')
        #print(f'weight:{net.weight.data},bias:{0}')
    pass
if __name__ == '__main__':
    freeze_support()
    main()