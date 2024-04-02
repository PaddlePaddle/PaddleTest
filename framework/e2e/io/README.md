# io模块API的功能测试框架
## 介绍
测试Paddle训练框架io模块API的功能测试。


## 模块功能
```
.    --------------------------> io模块API功能测试框架根目录
├── README.md
│
├── io_exec.py    ------------>  io生成器和测试执行器
│
├── io_loader.py  ------------>  Dataloader生成器
│
├── io_reader.py  ------------>  Dataset和Sampler生成器
│
├── io_test.py    ------------>  主测试框架，自测dataset和dataloader
│
├── io_trans.py   ------------>  yaml解析器，继承自weaktrans
│
└── dataloader.yml ----------->  case存放地址
```

## case描述
case是通过yaml文件描述，指定Dataset、Sampler、BatchSampler、Dataloader的相关参数。具体内容可参考： [io测试case](https://github.com/PaddlePaddle/PaddleTest/blob/develop/framework/e2e/io/dataloader.yml#:~:text=Blame-,DataGenerator0,-%3A)    <br>
可通过两种方式进行数据集定义：<br>
（1）定义数据集单元（一般为tensor结构），然后迭代该数据单元生成数据集label；<br>
（2）直接传入数据集（目前仅支持*.gz存储格式），进行数据集读取。


## 框架功能
（1）根据不同yaml描述的case产生dataset和dataloader：调用io_exec中GTCase实例化对象的generate方法；<br>
（2）对不同yaml描述的case所产生dataset和dataloader进行自测；对于大数据集自测采用抽样对照的方法：调用io_exec中GTCase实例化对象的run_case方法。<br>

## 执行
```
git clone https://github.com/PaddlePaddle/PaddleTest.git

cd PaddleTest/tree/develop/framework/e2e/io

python io_exec.py
```
