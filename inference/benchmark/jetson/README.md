# Benchmarks For Jetson

## 参考链接：

```shell
https://github.com/NVIDIA-AI-IOT/jetson_benchmarks
```

## 环境要求
* JetPack 4.4 / 4.5 / 4.6
* TensorRT 7 / 8
* cuda
* cudnn

## 前置条件
```shell
mkdir models # 创建模型文件路径
sudo sh install_requirements.sh #安装相关依赖库
```
## 下载模型
```shell
# Jetson NX
python3 utils/download_models.py --all --csv_file_path ./benchmark_csv/nx-benchmarks.csv --save_dir ./models
# Jetson AGX
python3 utils/download_models.py --all --csv_file_path ./benchmark_csv/xavier-benchmarks.csv --save_dir ./models
# Jetson TX2
python3 utils/download_models.py --all --csv_file_path ./benchmark_csv/tx2-nano-benchmarks.csv --save_dir ./models
```

## 运行单测(注意model 为绝对路径 否则会报错)
1.运行所有模型
```shell
#Jetson NX
sudo python3 benchmark.py --all --csv_file_path ./benchmark_csv/nx-benchmarks.csv --model_dir model绝对路径
#Jetson AGX
sudo python3 benchmark.py --all --csv_file_path ./benchmark_csv/xavier-benchmarks.csv \
                          --model_dir model绝对路径 \
                          --jetson_devkit xavier \
                          --gpu_freq 1377000000 --dla_freq 1395200000 --power_mode 0
#Jetson TX2
sudo python3 benchmark.py --all --csv_file_path ./benchmark_csv/tx2-nano-benchmarks.csv \
                            --model_dir 绝对路径 \
                            --jetson_devkit tx2 \
                            --gpu_freq 1122000000 --power_mode 3 --precision fp16
```
2.运行单个模型
```shell
#Jetson NX
sudo python3 benchmark.py --model_name inception_v4 --csv_file_path ./benchmark_csv/nx-benchmarks.csv --model_dir model绝对路径
#Jetson AGX
sudo python3 benchmark.py --model_name inception_v4  --csv_file_path ./benchmark_csv/xavier-benchmarks.csv \
                          --model_dir model绝对路径 \
                          --jetson_devkit xavier \
                          --gpu_freq 1377000000 --dla_freq 1395200000 --power_mode 0
#Jetson TX2
sudo python3 benchmark.py   --model_name inception_v4  --csv_file_path ./benchmark_csv/tx2-nano-benchmarks.csv \
                            --model_dir 绝对路径 \
                            --jetson_devkit tx2 \
                            --gpu_freq 1122000000 --power_mode 3 --precision fp16
```
##注意事项
* 由于 jetson tx2 和 nano 只有 GPU没有 DLA，结果获取的 FPS 即为 真实的GPU的 QPS
* 对于 jetson NX 和 AGX 存在 GPU和 DLA，结果获取的 FPS ： QPS （GPU）+ 2 ✖️ QPS（DLA ），因此需要修改单测文件只测试 GPU ：
```shell
# 1. 修改 jetson_benchmarks/utils/load_store_engine.py  12行
self.num_devices = 1 # 3 if GPU+2DLA, 1 if GPU Only
# 2. 修改 jetson_benchmarks/utils/read_write_data.py  17行
self.num_devices = 1 # data['Devices'][read_index]
```
*还可以将 jetson_benchmarks/utils/load_store_engine.py 151-152 行 注释，从而在 jetson_benchmarks/models/inception_v4_b4_ws2048_gpu.txt 查看详细的输出耗时、性能分析数据。
```shell
# 注释 jetson_benchmarks/utils/load_store_engine.py  151-152 行
# if os.path.isfile(_txtout_path):
#    os.remove(_txtout_path)
```

##Paddle GPU 测试步骤
1.采用 E2E 测试方式
```shell
time1 = time.time()
# core.nvprof_start()
# core.nvprof_enable_record_event()
for i in range(args.repeats):
    # core.nvprof_nvtx_push("forward " + str(i))
    input_tensor.copy_from_cpu(img[0].copy())
    predictor.run()
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
#     core.nvprof_nvtx_pop()
# core.nvprof_stop()
time2 = time.time()
total_inference_cost = (time2 - time1) * 1000  # total latency, ms
```
