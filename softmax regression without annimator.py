import torch 
from d2l import torch as d2l
from torch import nn
from multiprocessing import freeze_support
from torch.utils import data
from torchvision import transforms,datasets
def main():
    batch_size = 256
    data_form=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
    trainset=datasets.MNIST(root='./number_mnist',transform=data_form)
    testset=datasets.MNIST(root='./number_mnist',transform=data_form)
    train_iter=data.DataLoader(trainset,batch_size=batch_size,shuffle=False)
    test_iter=data.DataLoader(testset,batch_size=1,shuffle=True)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net[1].weight.data.normal_(0, 0.01)
    net[1].bias.data.fill_(0)
    loss = nn.CrossEntropyLoss()
    lr=0.2
#    def updater(batch_size):
#        return d2l.sgd(net.parameters(), lr, batch_size)
    trainer=torch.optim.SGD(net.parameters(),lr)
    num_epoch = 10
    for epoch in range(num_epoch):
        if isinstance(net, torch.nn.Module):
            net.train()
        for X, y in train_iter:
            l=loss(net(X),y)
#            if isinstance(updater, torch.optim.Optimizer):
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
#            else:
#                l.sum().backward()
#                updater(X.shape[0])
        net.eval()
        t=0
        with torch.no_grad():
            for X, y in test_iter:
                if t == 1:
                    break
                print(f'epoch:{epoch+1}, loss:{loss(net(X), y):f}')
                print(f'predict:{net(X).argmax(axis=1)},true:{y}')
                t=t+1
    pass
if __name__ == '__main__':
    freeze_support()
    main()