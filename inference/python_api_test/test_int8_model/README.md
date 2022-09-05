# Int8量化模型测试

## 准备数据&模型
```shell
sh prepare.sh
```

## Paddle Inference Tensor测试
- INT8

```shell
sh run_trt_int8.sh
```

- FP16

```shell
sh run_trt_fp16.sh
```

## Paddle Inference MKLDNN测试
```shell
sh run_mkldnn.sh
```
