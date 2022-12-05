export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE trt fp16
echo "[Benchmark] Run PPYOLOE trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco --reader_config=configs/ppyoloe_reader.yml --use_trt=True --precision=fp16 --model_name=PPYOLOE
# PPYOLOE trt fp16
echo "[Benchmark] Run PPYOLOE+ trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_plus_crn_s_80e_coco_no_nms --reader_config=configs/ppyoloe_plus_reader.yml --use_trt=True --precision=fp16 --model_name=PPYOLOE_PLUS --exclude_nms
# PicoDet trt fp16
echo "[Benchmark] Run PicoDet trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu --reader_config=configs/picodet_reader.yml --use_trt=True --precision=fp16 --model_name=PicoDet
# YOLOv5s trt fp16
echo "[Benchmark] Run YOLOv5s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_infer --use_trt=True --precision=fp16 --model_name=YOLOv5s
# YOLOv6s trt fp16
echo "[Benchmark] Run YOLOv6s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_infer --use_trt=True --precision=fp16 --model_name=YOLOv6s
# YOLOv7 trt fp16
echo "[Benchmark] Run YOLOv7 trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_infer --use_trt=True --precision=fp16 --model_name=YOLOv7

# ResNet_vd trt fp16
echo "[Benchmark] Run ResNet_vd trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_infer --use_trt=True --precision=fp16 --use_gpu=True --model_name=ResNet_vd
# MobileNetV3_large trt fp16
echo "[Benchmark] Run MobileNetV3_large trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_infer --use_trt=True --precision=fp16 --use_gpu=True --model_name=MobileNetV3_large
# PPLCNetV2 trt fp16
echo "[Benchmark] Run PPLCNetV2 trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_infer --use_trt=True --precision=fp16 --use_gpu=True --model_name=PPLCNetV2
# PPHGNet_tiny trt fp16
echo "[Benchmark] Run PPHGNet_tiny trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_infer --use_trt=True --precision=fp16 --use_gpu=True --model_name=PPHGNet_tiny
# EfficientNetB0 trt fp16
echo "[Benchmark] Run EfficientNetB0 trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_infer --use_trt=True --precision=fp16 --use_gpu=True --model_name=EfficientNetB0

# PP-HumanSeg-Lite trt fp16
echo "[Benchmark] Run PP-HumanSeg-Lite trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/ppseg_lite_portrait_398x224_with_softmax --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --use_trt=True --precision=fp16 --model_name=PP-HumanSeg-Lite
# PP-Liteseg trt fp16
echo "[Benchmark] Run PP-Liteseg trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-PPLIteSegSTDC1 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16 --model_name=PP-Liteseg
# HRNet trt fp16
echo "[Benchmark] Run HRNet trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-HRNetW18-Seg --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16 --model_name=HRNet
# UNet trt fp16
echo "[Benchmark] Run UNet trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-UNet --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16 --model_name=UNet
# Deeplabv3-ResNet50 trt fp16
echo "[Benchmark] Run Deeplabv3-ResNet50 trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-Deeplabv3-ResNet50 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=fp16 --model_name=Deeplabv3-ResNet50

# ERNIE 3.0-Medium trt fp16
echo "[Benchmark] Run ERNIE 3.0-Medium trt fp16"
$PYTHON test_nlp_infer.py --model_path=models/AFQMC --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --use_trt --precision=fp16 --model_name=ERNIE_3.0-Medium
# PP-MiniLM trt fp16
echo "[Benchmark] Run PP-MiniLM trt fp16"
$PYTHON test_nlp_infer.py --model_path=models/afqmc --task_name='afqmc' --use_trt --precision=fp16 --model_name=PP-MiniLM
# BERT Base trt fp16
echo "[Benchmark] Run BERT Base trt fp16"
$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola --use_trt --precision=fp16 --batch_size=1 --model_name=BERT_Base
