import torch
from torch import nn
from torch.utils import data
from torchvision import transforms,datasets
from multiprocessing import freeze_support
def main():
    batch_size=256
    data_form=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
    data_set=datasets.MNIST(root='./python from the scratch',transform=data_form)
    test_set=datasets.MNIST(root='./python from the scratch',transform=data_form)
    net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,10))
    train_iter=data.DataLoader(data_set,batch_size=batch_size,shuffle=True)
    test_iter=data.DataLoader(test_set,batch_size=batch_size,shuffle=True)
    loss=nn.CrossEntropyLoss(reduction='mean')
    trainer=torch.optim.SGD(net.parameters(),lr=0.1)
    num_epoch=10
    t=0
    def accuracy(y_hat,y):
        y_hat=y_hat.argmax(axis=1)
        cmp=(y_hat.type(y.dtype)==y)
        return float(cmp.type(y.dtype).sum()/len(y))
    for epoch in range(num_epoch):
        for X,y in train_iter:
            net.train()
            trainer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            trainer.step()
        net.eval()
        with torch.no_grad():
            for X,y in test_iter:
                print(f'epoch:{epoch+1},loss:{loss(net(X),y):f}')
#                print(f'predict:{net(X).argmax(axis=1)},label:{y}')
                print(f'accuracy_ratio={accuracy(net(X),y)}')
                t+=accuracy(net(X),y)
                break
    print(f'accruracy_ratio_accumulate={t/num_epoch:f}')
    pass
if __name__ == '__main__':
    freeze_support()
    main()