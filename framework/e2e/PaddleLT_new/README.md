# PaddleLayerTest子图测试框架（PLT）
__PLT旨在构建一个自动化测试框架，用于接入paddle.nn.Layer级别的子图case，用于评估、训练、动转静、预测等项目的精度、性能测试。__

## 代码文件结构

- generator 深度学习训练/评估的四大组成部分，即数据、模型、loss、优化器的构建
- engine 核心引擎，包含PaddlePaddle以及PyTorch框架的评估、训练、动转静、预测实现。
- db 数据库交互 ，各种数据库操作方法和基类
- layercase/layerApicase 子图模块，包含各种paddle.nn.Layer用例
- scene 环境变量配置，用于设置环境变量控制PLT执行过程中的各种开关
- strategy 精度、性能对比的评判标准与策略
- support 外围支持，包括子图py文件去重、yml配置转py文件等工具脚本
- tools 工具模块：发送邮件、加载yaml配置、保存pickle、上传产物等
- yaml 测试场景配置，控制核心引擎使用以及对比策略
- layertest.py 单个测试用例执行调试
- run.py 批量执行测试用例
- start.sh 启动docker容器并执行测试任务
