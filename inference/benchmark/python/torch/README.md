# Pytorch Inference Benchmark

## 镜像
```shell
docker pull nvcr.io/nvidia/pytorch:22.01-py3
```

## 相关依赖
```shell
python -m pip install opencv-python
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install --plugins
```

## 执行方式
```shell
python clas_benchmark.py --model_name resnet101 --device cpu --batch_size 1
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 1
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 1 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 1 --use_trt --trt_precision fp16
```
