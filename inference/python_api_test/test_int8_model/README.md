# Int8量化模型测试

## 准备数据&模型
```shell
sh prepare.sh
```

## Paddle Inference TensorRT测试
- INT8

```shell
sh run_trt_int8.sh > eval_trt_int8_acc.log 2>&1 &
```

收集重要log信息：
```shell
grep -i Benchmark eval_trt_int8_acc.log
```

- FP16

```shell
sh run_trt_fp16.sh > eval_trt_fp16_acc.log 2>&1 &
```

收集重要log信息：
```shell
grep -i Benchmark eval_trt_fp16_acc.log
```


## Paddle Inference MKLDNN测试
- INT8

```shell
sh run_mkldnn_int8.sh > eval_mkldnn_int8_acc.log 2>&1 &
```

收集重要log信息：
```shell
grep -i Benchmark eval_mkldnn_int8_acc.log
```

- FP32

```shell
sh run_mkldnn_fp32.sh > eval_mkldnn_fp32_acc.log 2>&1 &
```

收集重要log信息：
```shell
grep -i Benchmark eval_mkldnn_fp32_acc.log
```
