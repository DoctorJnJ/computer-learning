```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
```


```python
#生成数据集
true_w = torch.tensor([2, -3.4]) 
true_b = 4.2 
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
'''
等同于
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
'''
```


```python
#读取数据
def load_array(data_arrays, batch_size, is_train=True):      #data_arrays由张量组成的元组，包含了输入数据和标签数据
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```


```python
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```


```python
next(iter(data_iter))
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
```


```python
net = nn.Sequential(nn.Linear(2, 1))
```


```python
#初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```




    tensor([0.])




```python
#定义损失函数
loss = nn.MSELoss()
```


```python
#定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```


```python
#训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
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
