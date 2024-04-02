# GPU
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 1
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 4
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 8

python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 1
python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 4
python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 8

python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 1
python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 4
python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 8

python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 1
python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 4
python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 8

python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 1
python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 4
python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 8

# FP32
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 1 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 4 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 8 --use_trt --trt_precision fp32

python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 1 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 4 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 8 --use_trt --trt_precision fp32

python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 1 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 4 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 8 --use_trt --trt_precision fp32

python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 1 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 4 --use_trt --trt_precision fp32
python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 8 --use_trt --trt_precision fp32

python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 1 --use_trt --trt_precision fp32
python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 4 --use_trt --trt_precision fp32
python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 8 --use_trt --trt_precision fp32

# FP16
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 1 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 4 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name resnet101 --device gpu --batch_size 8 --use_trt --trt_precision fp16

python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 1 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 4 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name vgg16 --device gpu --batch_size 8 --use_trt --trt_precision fp16

python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 1 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 4 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name squeezenet1_0 --device gpu --batch_size 8 --use_trt --trt_precision fp16

python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 1 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 4 --use_trt --trt_precision fp16
python clas_benchmark.py --model_name mobilenet_v2 --device gpu --batch_size 8 --use_trt --trt_precision fp16

python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 1 --use_trt --trt_precision fp16
python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 4 --use_trt --trt_precision fp16
python detection_benchmark.py --model_name faster_rcnn --device gpu --batch_size 8 --use_trt --trt_precision fp16
