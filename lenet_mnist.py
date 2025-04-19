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
    net = nn.Sequential(
        nn.Conv2d(1,6,kernel_size=5,padding=2),nn.ReLU(),
        nn.AvgPool2d(kernel_size=2,stride=2),
        nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
        nn.AvgPool2d(kernel_size=2,stride=2),
        nn.Flatten(), nn.Linear(16*5*5, 120),nn.ReLU(),
        nn.Linear(120,84),nn.ReLU(),
        nn.Linear(84,10))
    def init_weight(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    loss = nn.CrossEntropyLoss()
    lr=0.2
#    def updater(batch_size):
#        return d2l.sgd(net.parameters(), lr, batch_size)
    trainer=torch.optim.SGD(net.parameters(),lr)
    num_epoch = 10
    lamda=1
    for epoch in range(num_epoch):
        if isinstance(net, torch.nn.Module):
            net.train()
        for X, y in train_iter:
            l=loss(net(X),y)+lamda*net[9].weight.data
#            if isinstance(updater, torch.optim.Optimizer):
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
#            else:
#                l.sum().backward()
#                updater(X.shape[0])
        net.eval()
        test_loss=0
        test_correct=0
        test_sum=0
        with torch.no_grad():
            for X, y in test_iter:
                test_sum+=1
                if net(X).argmax(axis=1)==y:
                    test_correct+=1
                test_loss+=loss(net(X),y)
            test_loss/=test_sum
            test_correct/=test_sum
            print(f'epoch:{epoch+1}, loss:{test_loss:f},test_accuracy:{test_correct:f}')
                # print(f'predict:{net(X).argmax(axis=1)},true:{y}')
    pass
if __name__ == '__main__':
    freeze_support()
    main()