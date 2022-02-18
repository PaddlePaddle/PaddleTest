#ResNet101
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=1 --use_gpu
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=4 --use_gpu
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=8 --use_gpu

#VGG16
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=1 --use_gpu
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=4 --use_gpu
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=8 --use_gpu

#MobileNetV2
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=1 --use_gpu
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=4 --use_gpu
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=8 --use_gpu

#ResNet101 tensorrt fp32
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=1 --use_gpu --use_trt --trt_precision=fp32
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=4 --use_gpu --use_trt --trt_precision=fp32
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=8 --use_gpu --use_trt --trt_precision=fp32

#VGG16 tensorrt fp32
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=1 --use_gpu --use_trt --trt_precision=fp32
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=4 --use_gpu --use_trt --trt_precision=fp32
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=8 --use_gpu --use_trt --trt_precision=fp32

#MobileNetV2 tensorrt fp32
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=1 --use_gpu --use_trt --trt_precision=fp32
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=4 --use_gpu --use_trt --trt_precision=fp32
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=8 --use_gpu --use_trt --trt_precision=fp32

#ResNet101 tensorrt fp16
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=1 --use_gpu --use_trt --trt_precision=fp16
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=4 --use_gpu --use_trt --trt_precision=fp16
python clas_keras_benchmark.py --model_name=ResNet101 --batch_size=8 --use_gpu --use_trt --trt_precision=fp16

#VGG16 tensorrt fp16
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=1 --use_gpu --use_trt --trt_precision=fp16
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=4 --use_gpu --use_trt --trt_precision=fp16
python clas_keras_benchmark.py --model_name=VGG16 --batch_size=8 --use_gpu --use_trt --trt_precision=fp16

#MobileNetV2 tensorrt fp16
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=1 --use_gpu --use_trt --trt_precision=fp16
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=4 --use_gpu --use_trt --trt_precision=fp16
python clas_keras_benchmark.py --model_name=MobileNetV2 --batch_size=8 --use_gpu --use_trt --trt_precision=fp16
