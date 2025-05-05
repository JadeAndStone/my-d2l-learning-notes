import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
from torch.utils import data
import torchvision
from torchvision import transforms,datasets
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import numpy as np
### data_process
data_file=".\Shenzhen Cup_data\Q1_numbers of people.xlsx"
features=pd.read_excel(data_file)
datas=features.iloc[:,1:92]

def process(data):
    str_allele='Allele '
    for i in range(30):
        data=data.drop(str_allele+str(i+1),axis=1)
    numeric=data.dtypes[data.dtypes!='object'].index
    data[numeric]=data[numeric].apply(lambda x: (x-x.mean())/(x.std()))
    data[numeric]=data[numeric].fillna(0)
    data=pd.get_dummies(data)
    return data
datas=process(datas)
datas=torch.tensor(datas.values,dtype=torch.float32)
pca=PCA(n_components="mle")
# print(pca.fit_transform(datas).shape)
datas=pca.fit_transform(datas)
datas=torch.tensor(datas,dtype=torch.float32)
label=torch.zeros(datas.shape[0])

for i in range(15*16):
    label[i]=0
for i in range(15*16,31*16,1):
    label[i]=1
for i in range(31*16,42*16,1):
    label[i]=2
for i in range(42*16,51*16,1):
    label[i]=3
### net
# net = nn.Sequential(
#     nn.Linear(datas.shape[-1],256),
#     nn.BatchNorm1d(256),
#     nn.ReLU(),
#     # nn.Dropout(0.2),
    
#     nn.Linear(256,64),
#     nn.BatchNorm1d(64),
#     nn.ReLU(),
#     # nn.Dropout(0.5),
    
#     nn.Linear(64,4)
# )
net = nn.Sequential(
        nn.Conv1d(1,16,kernel_size=2,stride=2),nn.ReLU(),
        nn.MaxPool1d(kernel_size=2,stride=2),
        nn.Conv1d(16,32,kernel_size=2),nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(32,64,kernel_size=2),nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(64,128,kernel_size=2),nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(128,4)
)
## train
loss=nn.CrossEntropyLoss(reduction='mean')
params = [
    {"params":net[0].parameters(),"weight_decay":0.01},
    {"params":net[3].parameters(),"weight_decay":0.03},
    {"params":net[6].parameters(),"weight_decay":0.08},
    {"params":net[9].parameters(),"weight_decay":0.15}
]
trainer=torch.optim.AdamW(net.parameters(),lr=0.003)
# trainer=torch.optim.SGD(net.parameters(),lr=0.2,momentum=0.9)

def iter(data_array,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
KF=KFold(n_splits=5)
batch_size=32
num_epoch=50

index=[i for i in range(datas.shape[0])]
np.random.shuffle(index)
datas,label=datas[index],label[index]
datas=datas.reshape((-1,1,68))
def init_weight(m):
    if type(m)==nn.Linear or type(m)==nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
def accuracy(y_hat,y):
    y_hat=y_hat.argmax(axis=1)
    cmp=(y_hat.type(y.dtype)==y)
    return float(cmp.type(y.dtype).sum()/len(y))
final_test_accuracy=0
final_accuracy=0
for train_index,test_index in KF.split(datas):
    train_iter=iter((datas[train_index],label[train_index]),batch_size=batch_size)
    test_iter=iter((datas[test_index],label[test_index]),batch_size=batch_size)
    net.apply(init_weight)
    for epoch in range(num_epoch):
        net.train()
        for X,y in train_iter:
            trainer.zero_grad()
            l=loss(net(X),y.long())
            l.backward()
            trainer.step()
        net.eval()
        with torch.no_grad():
            print(f'epoch:{epoch+1} train_loss:{loss(net(datas[train_index]),label[train_index].long()):f} train_accuracy:{accuracy(net(datas[train_index]),label[train_index])}')
            print(f'epoch:{epoch+1} test_loss:{loss(net(datas[test_index]),label[test_index].long()):f} test_accuracy:{accuracy(net(datas[test_index]),label[test_index])}')
    net.eval()
    with torch.no_grad():
        final_test_accuracy+=accuracy(net(datas[test_index]),label[test_index])
        final_accuracy+=accuracy(net(datas),label)
print(f'average_test_accuracy:{final_test_accuracy/5}')
print(f'average_accuracy:{final_accuracy/5}')