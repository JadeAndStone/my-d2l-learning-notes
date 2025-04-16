import torch
from torch import nn
import pandas as pd
import os
import hashlib
import requests
import zipfile
import tarfile
from torch.utils import data
from multiprocessing import freeze_support
from sklearn.model_selection import KFold
def main():
    DATA_HUB=dict() #用来存文件名（key）和其对应的url和验证文件是否完整的shal密钥（value）
    DATA_URL='http://d2l-data.s3-accelerate.amazonaws.com/'
    def download(name,cache_dir=os.path.join('..','kaggle_data')):
        #join安全合并两个路径，'..'是当前文件所在目录的上一级目录,
        #cache_dir即为'..'下的data文件夹路径
        assert name in DATA_HUB,f"{name}didn't exist in{DATA_HUB}"
        url,shal_hash=DATA_HUB[name]
        os.makedirs(cache_dir,exist_ok=True)#exist_ok=True目录已存在时不报错
        fname=os.path.join(cache_dir,url.split('/')[-1])
        #这里url.split('/')是以url字符串中的'/'作为分隔符实现对url字符串的分割
        #返回多段字符串，在这里是每一级文件夹名，[-1]选取其中的最后一个字符串
        #也就是文件的名字，所以合并后的fname就是每个文件的路径
        if os.path.exists(fname):#判断fname路径的文件是否存在
            sha1=hashlib.sha1()#创建一个哈希计算对象，下面进行哈希验证
            with open(fname,'rb') as f:#以二进制('b')模式读入('r')fname处的文件
                while True:
                    data=f.read(1048576)#这是1MB的字节数，每次读取1MB，防止文件过大内存溢出
                    if not data:#如果文件读完了，data就是空的，退出
                        break
                    sha1.update(data)#哈希更新
            if sha1.hexdigest()==shal_hash:
                #如果sha1与文件预存的hash（即shal密钥）一致，
                #说明文件完整，直接返回路径
                return fname
        print(f'正在从{url}下载{fname}...')
        r=requests.get(url,stream=True,verify=True)
        #网络请求初始化，访问网站文件中的数据
        with open(fname,'wb') as f:#二进制写入模式
            f.write(r.content)
            #写入网站对应文件的内容（存在则覆盖，不存在则直接写）
            #用来补全不完整的文件
        return fname
    def download_extract(name,folder=None):
        fname=download(name)
        base_dir=os.path.dirname(fname)#返回上一级目录，这里是为了让文件解压到当前文件夹下
        data_dir,ext=os.path.splitext(fname)#将fname拆分为基础路径和文件后缀
        if ext=='.zip':
            fp=zipfile.ZipFile(fname,'r')#选择zip的解压器
        elif ext in ('.tar','gz'):
            fp=tarfile.open(fname,'r')#选择tar的解压器
        else:
            assert False,'只有zip/tar文件可以被解压缩'
        fp.extractall(base_dir)#解压到base_dir
        return os.path.join(base_dir,folder)if folder else data_dir
        #如果folder存在的话就返回base与folder的结合路径（如果有多个文件夹的话可以指定folder）
        #如果不存在的话就返回data_dir，即解压完成后文件的路径
    def download_all():
        for name in DATA_HUB:
            download(name)
    DATA_HUB['kaggle_house_train'] = (
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    train_data=pd.read_csv(download('kaggle_house_train'))
    test_data=pd.read_csv(download('kaggle_house_test'))
    all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
    numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
    #这里all_features.dtypes!='object'对all_features中每一列判断是否为object类型
    #返回对应的布尔掩码，进而提取出all_features中的非object列，再用index
    #就返回了all_features中的数字类型列的列标
    all_features[numeric_features]=all_features[numeric_features].apply(
        lambda x: (x-x.mean())/(x.std())
        #这里对数字类型列的数据进行归一化
        #运用lambda函数——简短的内联函数，可对大量x进行统一操作
    )
    all_features[numeric_features]=all_features[numeric_features].fillna(0)
    #归一化之后每一行均值都化为0，故可将nan值替换为0
    all_features=pd.get_dummies(all_features,dummy_na=True)
    #这里将数据的非数字类型的列转换成独热编码，即将某一列中的一种字符串单独
    #拿出来创建一列，用0，1表示某一行是不是这种字符串，这里将na值也单独
    #拿了出来
#    all_features=all_features.apply(lambda x: (x-x.mean())/(x.std()))
#    all_features=all_features.fillna(0)
#    Guass=torch.randn_like(all_features)
#    all_features+=Guass
    n_train=train_data.shape[0]
    train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float32)
    test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float32)
    #这里利用.value从pandas中提取出numpy格式的数据，并将其转化为tensor
    train_labels=torch.tensor(train_data.SalePrice.values,dtype=torch.float32)
    #这里用.iloc[:,-1]也可以，不过直接查询列名更泛用
    net = nn.Sequential(
        nn.Linear(train_features.shape[-1], 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),

        nn.Linear(64, 32),
        nn.ReLU(),

        nn.Linear(32, 1)
    )
    batch_size=8
#    for i in range(2):
    print(train_features[1],train_labels[1])
    def data_iter(data_array,is_train=True):
        dataset=data.TensorDataset(*data_array)
        return data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train)
    train_labels_normal=(train_labels-train_labels.mean())/(train_labels.std())
    test_iter=data_iter(test_features)
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)
#    print(net[0].weight.data)
    net[7].weight.data.normal_(0,0.01)
    net[7].bias.data.fill_(0)
    net[4].weight.data.normal_(0,0.01)
    net[4].bias.data.fill_(0)
    net[9].weight.data.normal_(0,0.01)
    net[9].bias.data.fill_(0)
    lr=0.001
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.HuberLoss(reduction='mean',delta=1)
    M_loss=nn.MSELoss(reduction='mean')
    num_epoch=10
    lamda=2
    KF=KFold(n_splits=5)
    for epoch in range(num_epoch):
        t=0
        for train_index,test_index in KF.split(train_features):
            t+=1
            train_features_temp,train_labels_temp=train_features[train_index],train_labels[train_index]
            test_features_temp,test_labels_temp=train_features[test_index],train_labels[test_index]
            train_iter=data_iter((train_features_temp,train_labels_temp.log()))
            test_iter=data_iter((test_features_temp,test_labels_temp.log()))
            for X,y in train_iter:
                net.train()
                trainer.zero_grad()
                l=loss(net(X),y)+lamda*torch.norm(net[0].weight.data)**2
                l.backward()
                trainer.step()
#            if t==0:
#                t+=1
#                print(lr*net[0].weight.grad)
            print(net(train_features).std(),(train_labels).log().std())
#        print(net[0].weight.data,net[0].bias.data)
#        print(torch.matmul(train_features[0:10],net[0].weight.data.T)+net[0].bias.data)
#        print(train_labels[0:10]/10000)
#        print(net(train_features).mean(),(train_labels/10000).mean())
            net.eval()
            with torch.no_grad():
                print(f'epoch:{epoch+1},fold:{t},loss:{loss(net(train_features_temp),train_labels_temp.log()):f}')
                #print((abs((net(test_features_temp)/(test_labels_temp.log()))-1)).mean())
                print((abs((net(test_features_temp).exp()/(test_labels_temp))-1)).mean())
    pass
if __name__ == '__main__':
    freeze_support()
    main()