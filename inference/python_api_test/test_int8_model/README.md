
# 测试须知

为了保证你测试的性能、显存数据很准确，你必须保证你独占了那张GPU卡！
测试前，请用nvidia-smi命令查看下，保证你测试的那张卡的显存占用量为0，防止一些暂停的进程还在占用显存。

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
