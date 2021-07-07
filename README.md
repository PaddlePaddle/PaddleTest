# PaddleTest - PaddlePaddle测试套件
目录结构
```
.    -------------------------------------> 项目根目录
├── README.md
│
├── distributed/ -------------------------> 分布式测试case
│
├── framework/   -------------------------> 主框架Paddle测试case
│
├── inference/   -------------------------> 预测相关测试case
│
├── lite/        -------------------------> Lite相关测试case
│
└── models/      -------------------------> 模型相关的测试case
```


# 如何贡献
## 代码审查
代码审查包括两部分，一个是基于现有开源代码格式规范的审查，通过pre-commit来审查，另一个是基于自定义要求的审查，在`tools/codestyle`下。

pre-commit审核主要是三种，包括`black`,`flake8`,`pylint`，在CI阶段代码审查执行。本地运行方式是安装pre-commit，一个简单的方法是用python3+使用`pip install pre-commit`来安装。
执行方式`pre-commit run --file [your code file]`，`black`会自动调整代码格式和一些简单的规范错误。具体规范配置请见根目录`.pre-commit-config.yaml`文件。

## 合入规范
合入**必须**要求通过全部CI检测，原则上禁止强行Merge，如果有Pylint代码格式阻塞，可以讨论是否禁止某一条规范生效，**必须**要求一个QA Reviewer，禁止出现敏感代码。
