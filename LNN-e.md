```python
import numpy as np
#NumPy 库导入到当前代码中，并将其命名为 np
import torch
#将 PyTorch 库导入
from torch.utils import data
#将 PyTorch 库中 utils 模块中的data 子模块导入
from d2l import torch as d2l
#d2l.torch 是 Dive into Deep Learning  Python 包中的子模块，提供了使用 PyTorch 进行深度学习的工具和函数
```


```python
#生成数据集
true_w = torch.tensor([2, -3.4]) 
true_b = 4.2 
features, labels = d2l.synthetic_data(true_w, true_b, 1000)   #定义的synthetic_data函数如下所示
'''
等同于
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    #均值为0，标准差为1正态分布中随机抽取，张量为num_examples 行和 len(w) 列
    y = torch.matmul(X, w) + b
    #torch.matmul()计算两个张量的矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    #添加高斯噪声,随机数是从均值为 0，标准差为 0.01 的正态分布中抽取
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
'''
```


```python
#读取数据
def load_array(data_arrays, batch_size, is_train=True):  #data_arrays由张量组成的元组，包含输入数据和标签数据 data_arrays = (features, labels)
    dataset = data.TensorDataset(*data_arrays)
    #data_arrays 表示将 data_arrays 中的所有数组或张量解包，作为单独的参数传递给 data.TensorDataset() 函数
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```


```python
batch_size = 10
#指定每次训练的批次大小，从训练数集中随机选取的数据样本数量
data_iter = load_array((features, labels), batch_size)
#数据迭代器，将数据集按照批次大小batch_size进行分批处理
```


```python
next(iter(data_iter))
#next()函数则从迭代器对象中获取下一个元素，即下一个批次的数据
```




    [tensor([[-1.1038,  2.2042],
             [-1.4543,  0.0129],
             [-0.4661,  0.3747],
             [ 0.1385,  0.5418],
             [ 1.2324,  0.1342],
             [-0.0533,  0.1338],
             [-1.4412,  0.7663],
             [ 1.4620, -0.0172],
             [ 1.2864, -0.5398],
             [-0.4108, -1.2307]]),
     tensor([[-5.4923],
             [ 1.2318],
             [ 1.9805],
             [ 2.6478],
             [ 6.1871],
             [ 3.6422],
             [-1.3147],
             [ 7.1666],
             [ 8.6086],
             [ 7.5672]])]




```python
#定义模型
from torch import nn
#将 PyTorch 库中的 nn 模块导入,nn模块提供常用的层、激活函数、损失函数等工具和类

```


```python
net = nn.Sequential(nn.Linear(2, 1))
'''
Sequential 是 PyTorch 中的一个类，它是一个容器，用于按顺序组合多个神经网络层。
使用 Sequential 类可以方便地将多个层组合成一个神经网络模型。
Sequential 的常见用法是将多个层按顺序添加到容器中，然后将容器作为一个整体使用。
'''
```


```python
#初始化模型参数
net[0].weight.data.normal_(0, 0.01)
#对神经网络模型中的第一个层（即索引为 0 的层）的权重进行初始化
net[0].bias.data.fill_(0)
#对神经网络模型中的第一个层的偏置进行初始化，将其全部设置为 0
```




    tensor([0.])




```python
#定义损失函数
loss = nn.MSELoss()
#均方误差损失，MSE 损失函数通常用于回归问题
```


```python
#定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
#net.parameters()是一个函数调用语句，它的作用是返回神经网络net中所有可学习的参数，以便进行优化器的更新操作
```


```python
#训练
#训练
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
    #在每一轮训练中，使用数据迭代器data_iter分批读取训练数据X和标签y
        l = loss(net(X) ,y)
        #net.parameters()是一个函数调用语句，它的作用是返回神经网络net中所有可学习的参数，以便进行优化器的更新操作
        trainer.zero_grad()
        l.backward()
        trainer.step()
        #使用优化器trainer对神经网络中的可学习参数进行梯度反向传播和更新操作
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    #打印当前轮次的训练结果，包括轮次编号和损失函数值

```

    epoch 1, loss 0.000232
    epoch 2, loss 0.000098
    epoch 3, loss 0.000097
    


```python
#误差
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```

    w的估计误差： tensor([ 0.0004, -0.0003])
    b的估计误差： tensor([7.4863e-05])
    


```python

```


```python

```
