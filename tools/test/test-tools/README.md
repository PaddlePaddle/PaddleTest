

# tool-check-availability-of-installation（安装验证工具）


#### 介绍


成功安装是深度学习框架能够正常使用的前提，尤其是大部分深度学习框架都具有安装包体积大、依赖多的特点，所以是否能够方便快捷的安装使用，对于用户非常重要，该工具用以验证深度学习安装框架的安装步骤，包括执行了工具的具体安装，以及安装后的简单组网和GPU执行方式可行性验证。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |              取值              |  描述  |
| :--------: | :----: | :------: | :----------------------------: | :----: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| check_item | string |    是    | simple_network、cuda_available | 检查项 |


正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```





# tool-test-dl-net（组网功能验证工具）


#### 介绍


组网构造神经网络是深度学习框架的核心基本功能。深度学习的网络层可以看成一个复杂的非线性方程组，这个复杂的非线性方程组经过AI工程师的设计，呈现出一层一层的结构。数据输入这个复杂的非线性方程组，经过层层计算，最终可以得出人们想要的结果（人脸识别，目标检测，机器人对话）。该工具可以用来展示模型的网络层的抽象结构，以及模型每一层形状，同时支持3个常用框架的模型结构展示。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
|net| string|   是    | vgg16, resnet101 |网络结构 |



正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```




# tool-test-save-load（模型加载和保存测试工具）


#### 介绍

模型加载是推理预测的前提，能够获取已经训练好的模型参数与结构，模型保存是训练完成后的重要步骤，能够对训练优化的模型参数与结构进行格式化存储，本工具用以验证框架保存与加载模型功能的可用性，包括自定义加载与保存参数与结构，支持3个常用框架的加载与保存功能。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
|action|string|是|save , load|操作(保存/加载)|
|content|string|是|net , params , model|内容|


正常响应


```
{"status": 200, "msg": "save/load path", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```



# tool-test-cpu-train（CPU训练验证工具）


#### 介绍


训练是深度学习的核心流程，一套完整的深度学习训练系统包含以下4大组成部分：1、数据加载器。2、模型网络结构。3、损失函数。4、优化器学习率。数据加载器接收输入的数据，传入模型网络结构中，在网络最后通过损失函数计算出损失值，然后将损失值结合优化器学习率更新模型网络结构这一复杂的非线性方程组中的每一个参数。循环上述过程，最终损失值会收敛到一个不变的范围，此时可以认为训练完成了。该工具用以展示一些基础模型的CPU训练功能。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model_name| string|   是    | vgg16, resnet101|模型名称 |



正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```








# tool-test-gpu-train（GPU训练验证工具）


#### 介绍


深度学习领域由于计算量大经常使用GPU训练。GPU机器擅长于矩阵数组计算，运用其特有的计算方式，训练的速度会比CPU快很多。PaddlePaddle可以非常便利地从CPU切换至GPU进行训练，而不需要繁琐地更改训练的底层代码。相对于CPU计算结果，GPU的计算结果会有一些差异，这主要是因为GPU高性能的计算方式造成了差异。使用该工具用以展示一些基础模型的GPU训练功能。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| net| string|   是    | vgg16, resnet101|模型名称 |



正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```


#### 示例








# tool-test-inference（端到端部署推理测试工具）


#### 介绍


推理部署是模型训练完成后，投入业务使用的重要环节，因而各个框架都提供了推理部署功能，使用预先训练的模型，接收输入经过计算后输出推理结果，本工具重点关注通过框架训练后的模型，能够正常部署，进行推理预测，同时支持3个常用框架的经典模型推理预测。







请求参数（Query）：


| 名称  |  类型  | 是否必须 |        取值         |   描述   |
| :---: | :----: | :------: | :-----------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model | string |    是    | mobilenet、resnet50 | 模型名称 |


正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```








# tool-test-op-correctness（算子正确性验证工具）


#### 介绍

算子是神经网络组网的基本单元，算子的正确性直接决定了最终网络及模型的正确性和准确度。算子正确性验证工具支持测试不同框架下，指定算子的前向及反向的正确性，并输出运行结果。






请求参数（Query）：


|         名称         |  类型   | 是否必须 |          取值           |        描述        |
| :------------------: | :-----: | :------: | :-----------------------: | :----------------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
|        op_name        | string  |    是    |        relu6、tanh、Sigmoid        |        op名称        |




正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```







# tool-test-train-performance（训练耗时测试工具）


#### 介绍
由于神经网络越来越复杂，而深度学习通常需要使用大量数据来进行训练，因而训练耗时成为了影响模型训练和调优的重要因素。本工具通过在训练过程中记录消耗的时间，同时记录处理过的图片数量，计算得出IPS指标数据（每秒钟处理的图像数量）。支持了Paddle、PyTorch、TensorFlow3个主流框架以及单卡和多卡训练场景。







请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model_name | string | 是 | paddle: yolov3<br>pytorch: gpt<br>tensorflow: mnist | 模型名称 |
| cards | number | 是 | 1, 8 | GPU数 |


正常响应


```
{"msg":"","result":"{\"ips\": 47582.722}","status":200}
```


失败响应


```
{"msg":"error message","result": "FAIL"，"status": 500}
```






# tool-test-train-resource（资源消耗测试工具）


#### 介绍
为了加速模型训练，从而可以减少迭代和调优时间成本，深度学习领域通常使用AI芯片进行加速。而深度学习框架能够把资源使用起来，是资源是否有效利用的重要指标，同时也可以在模型迭代过程中，衡量资源占用情况是否稳定，避免引入资源不足或性能下降等问题。本工具在训练过程中，监控了CPU利用率、最大内存占用、最大显存占用、GPU利用率4个指标。








请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model_name | string | 是 | paddle: yolov3<br>pytorch: gpt<br>tensorflow: mnist | 模型名称 |
| cards | number | 是 | 1, 8 | GPU数 |


正常响应


```
{"msg":"","result":"{avg_cpu_util: 0.20, max_gpu_memory_usage_mb: 5017, avg_gpu_util: 0.86}","status":200}
```


失败响应


```
{"msg": "error message"，"result": "FAIL"，"status":500,}
```





# tool-test-dl-algorithm-convergence（算法正确性验证工具）


#### 介绍
算法是深度学习领域为解决问题而设计的一系列运算，最终输出推理结果，而算法的正确性是算法价值的最根本的评价指标，本工具重点关注对于给定的输入，经过算法计算后，得出的结果是否在精度误差之内。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model_name | string | 是 | PaddlePaddle: resnet50, vgg11, yolov3, bert, ernie, fasterrcnn, mobilenetv1, ocr, deeplabv3, maskrcnn<br>Pytorch: resnet50, vgg16<br>TensorFlow: resnet50, vgg16 | 模型名称 |


正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```










# tool-test-dl-algorithm-performance（算法性能测试工具）


#### 介绍
深度学习框架通过组网构建一定结构的模型，来实现相应的深度学习算法，这也正是框架的使用场景，能够正确的表达算法并且通过算法的计算得到正确的结果，是深度学习框架的基本功能，本工具主要关注算法计算性能指标，在经过warm up后进行重复执行，并统计平均时延和QPS数据。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model_name | string | 是 | PaddlePaddle: AlexNet, ResNet50, DarkNet53, MobileNetV1, EfficientNetB0, MobileNetV2<br>Pytorch: resnet50, vgg16<br>TensorFlow: resnet50, vgg16 | 模型名称 |


正常响应


```
{"status": 200, "msg": "output", "result": {"avg_latency_ms": "7.067", "qps": "141.501"}}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```






# tool-test-dygraph-and-static-train (训练方式验证工具)


#### 介绍
深度学习训练方式的计算图分为两种，静态图和动态图。动态图意味着计算图的构建和计算同时发生。这种机制由于能够实时得到中间结果的值，使得调试更加容易，对于编程实现来说更友好。静态图则意味着计算图的构建和实际计算是分开的。在静态图中，会事先了解和定义好整个运算流，这样之后再次运行的时候就不再需要重新构建计算图了，因此速度会比动态图更快，从性能上来说更加高效。本工具是用来验证不同的深度学习框架是否支持动态图或静态图训练方式，并能够正确性进行模型训练。





请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| train_mode| string|     是  | static,、dygraph |静态图, 动态图 |


正常响应


```
{"status": 200, "msg": "output", "result": "PASS"}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```






# tool-test-inference-performance（并发推理性能测试工具）


#### 介绍
训练产出的模型在投入生产时，通常用于在线并发场景，这时需要关注深度学习框架的并发推理性能，本工具支持输入并发线程数，在并发请求场景下监控推理耗时和输入数量，进而计算并返回平均时延和QPS数据。




请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model_name | string | 是 | PaddlePaddle: AlexNet, ResNet50, DarkNet53, MobileNetV1, EfficientNetB0, MobileNetV2<br>Pytorch: resnet50, vgg16<br>TensorFlow: resnet50, vgg16 | 模型名称 |
| thread_num | number | 是 | 1、4 | 线程数 |


正常响应


```
{"status": 200, "msg": “output”, "result": {"avg_latency_ms": "9.1662437915802", "qps": "109.09594188609363"}}
```


失败响应


```
{"status": 500, "msg": "error message", "result": "FAIL"}
```






# tool-test-op-performance（算子性能测试工具）


#### 介绍
深度学习的性能的核心评价指标之一就是计算速度。算子的运行速度决定了从模型训练到预测部署推理一系列过程的效率。本工具目的是来验证执行算子在一轮前反向计算流中的执行时间，在多种深度学习框架之间进行对比。通过同一个算子在不同深度学习框架下的执行时间来反映出不同深度学习框架的运行性能。





请求参数（Query）：


|         名称         |  类型   | 是否必须 |          取值           |        描述        |
| :------------------: | :-----: | :------: | :-----------------------: | :----------------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
|        op_name        | string  |    是    |        relu、sigmoid        |        op名称     |




正常响应


```
{"status": 200, "msg": "output", "result": {"avg_latency_ms": "9.1662"}
```


失败响应


```
{"status": 500, "msg": "error msg", "result": "FAIL"}
```




# tool-test-dl-algorithm-convergence（算法收敛性测试工具）


#### 介绍
算法模型结构设计后，在训练迭代过程中，loss能否收敛，是能否成功产出预期模型的重要衡量指标。本工具监控在模型训练中loss值的变化，返回loss值，从而衡量该算法呢是否收敛。支持了Paddle、PyTorch、TensorFlow3个主流框架以及单卡和多卡训练场景。







请求参数（Query）：


|    名称    |  类型  | 是否必须 |                            取值                            |   描述   |
| :--------: | :----: | :------: | :----------------------------------------------------------: | :------: |
| framework  | string |    是    |  paddle、pytorch、tensorflow   | 框架名称 |
| model_name | string | 是 | paddle: xlnet<br>pytorch: gpt<br>tensorflow: mnist | 模型名称 |
| cards | number | 是 | 1、8 | GPU卡数 |


正常响应


```
{"msg":"","result":"{\"value\": 2.671735E+04, \"base_value\": 2.52210E+04, \"gap\": 0.059329526981483624}","status":200}
```


失败响应


```
{"msg":"error message", "result":""，"status":500}
```


