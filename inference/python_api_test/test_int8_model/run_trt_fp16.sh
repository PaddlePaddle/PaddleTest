export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE trt fp16
echo "[Benchmark] Run PPYOLOE trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco --reader_config=configs/ppyoloe_reader.yml --use_trt=True --precision=fp16
# PicoDet trt fp16
echo "[Benchmark] Run PicoDet trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu --reader_config=configs/picodet_reader.yml --use_trt=True --precision=fp16
# YOLOv5s trt fp16
echo "[Benchmark] Run YOLOv5s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_infer --use_trt=True --precision=fp16 --arch=YOLOv5
# YOLOv6s trt fp16
echo "[Benchmark] Run YOLOv6s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_infer --use_trt=True --precision=fp16 --arch=YOLOv6
# YOLOv7 trt fp16
echo "[Benchmark] Run YOLOv7 trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_infer --use_trt=True --precision=fp16 --arch=YOLOv7

# ResNet_vd trt fp16
echo "[Benchmark] Run ResNet_vd trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_infer --use_trt=True --use_fp16=True --use_gpu=True
# MobileNetV3_large trt fp16
echo "[Benchmark] Run MobileNetV3_large trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_infer --use_trt=True --use_fp16=True --use_gpu=True
# PPLCNetV2 trt fp16
echo "[Benchmark] Run PPLCNetV2 trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_infer --use_trt=True --use_fp16=True --use_gpu=True
# PPHGNet_tiny trt fp16
echo "[Benchmark] Run PPHGNet_tiny trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_infer --use_trt=True --use_fp16=True --use_gpu=True
# EfficientNetB0 trt fp16
echo "[Benchmark] Run EfficientNetB0 trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_infer --use_trt=True --use_fp16=True --use_gpu=True

# PP-HumanSeg-Lite MKLDNN fp16
echo "[Benchmark] Run PP-HumanSeg-Lite MKLDNN fp16"
$PYTHON test_segmentation_infer.py --model_path=models/ppseg_lite_portrait_398x224_with_softmax --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --use_trt=True --precision=fp16
# PP-Liteseg MKLDNN fp16
echo "[Benchmark] Run PP-Liteseg MKLDNN fp16"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-PPLIteSegSTDC1 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16
# HRNet MKLDNN fp16
echo "[Benchmark] Run HRNet MKLDNN fp16"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-HRNetW18-Seg --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16
# UNet MKLDNN fp16
echo "[Benchmark] Run UNet MKLDNN int8"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-UNet --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16
# Deeplabv3-ResNet50 MKLDNN fp16
echo "[Benchmark] Run Deeplabv3-ResNet50 MKLDNN fp16"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-Deeplabv3-ResNet50 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16

# ERNIE 3.0-Medium trt fp16
echo "[Benchmark] Run ERNIE 3.0-Medium trt fp16"
$PYTHON test_nlp_infer.py --model_path=models/AFQMC --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --use_trt --precision=fp16
# PP-MiniLM trt fp16
echo "[Benchmark] Run PP-MiniLM trt fp16"
$PYTHON test_nlp_infer.py --model_path=models/afqmc --task_name='afqmc' --use_trt --precision=fp16
# BERT Base trt fp16
echo "[Benchmark] Run BERT Base trt fp16"
$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola --use_trt --precision=fp16 --batch_size=1
