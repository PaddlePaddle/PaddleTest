# API算子新自动微分测试框架
## 介绍
测试训练框架API的高阶自动微分精度，目前仅支持Paddle静态图。

## 模块功能
```
.    ------------------------> api算子新AD测试框架根目录
├── README.md
│
├── generate.py ------------>  测试case生成器
│
├── gradrun.py  ------------>  测试case执行器
│
├── gradtrans.py ----------->  yaml解析器，继承自weaktrans
│
├── competitive.py --------->  主测试框架，对比算子微分精度
│
├── tool.py     ------------------>  工具函数和类
│
├── start.py    ------------------>  自动化执行脚本
│
└── ../yaml/  ----------------->  case存放地址
```

## case描述
case是通过yaml文件描述，内容包括paddle和Jax的对比信息。[存放位置](https://github.com/PaddlePaddle/PaddleTest/tree/develop/framework/e2e/yaml)
具体内容可参考： [API竞品测试case描述](https://github.com/PaddlePaddle/PaddleTest/tree/develop/framework/e2e/competitor#case%E6%8F%8F%E8%BF%B0)

## 测试内容
不同初始梯度下：
① Paddle反向自动微分高阶精度，与Jax对齐，测试5阶；
② Paddle前向自动微分高阶精度，与Jax对齐，测试5阶；
③ Paddle反向自动微分与前向自动微分高阶精度对齐，测试5阶。
## 执行

```
git clone https://github.com/PaddlePaddle/PaddleTest.git

cd PaddleTest/tree/develop/framework/e2e/autograd

python generate.py base
python test_base.py base
```
