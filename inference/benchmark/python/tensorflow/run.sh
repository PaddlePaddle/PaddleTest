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


