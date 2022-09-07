export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE MKLDNN
echo "[Benchmark] Run PPYOLOE MKLDNN fp32"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --device=CPU --use_mkldnn=True --cpu_threads=10
# PicoDet MKLDNN
echo "[Benchmark] Run PicoDet MKLDNN fp32"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml
# YOLOv5s MKLDNN
echo "[Benchmark] Run YOLOv5s MKLDNN fp32"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv5
# YOLOv6s MKLDNN
echo "[Benchmark] Run YOLOv6s MKLDNN fp32"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv6
# YOLOv7 MKLDNN
echo "[Benchmark] Run YOLOv7 MKLDNN fp32"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv7

# ResNet_vd MKLDNN
echo "[Benchmark] Run ResNet_vd MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True
# MobileNetV3_large MKLDNN
echo "[Benchmark] Run MobileNetV3_large MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True
# PPLCNetV2 MKLDNN
echo "[Benchmark] Run PPLCNetV2 MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True
# PPHGNet_tiny MKLDNN
echo "[Benchmark] Run PPHGNet_tiny MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT --cpu_num_threads=10e --use_mkldnn=True --eval=True
# EfficientNetB0 MKLDNN
echo "[Benchmark] Run EfficientNetB0 MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True

# PP-HumanSeg-Lite MKLDNN fp32
echo "[Benchmark] Run PP-HumanSeg-Lite MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/ppseg_lite_portrait_398x224_with_softmax --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --device=CPU --use_mkldnn=True --cpu_num_threads=10
# PP-Liteseg MKLDNN fp32
echo "[Benchmark] Run PP-Liteseg MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-PPLIteSegSTDC1 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_num_threads=10
# HRNet MKLDNN fp32
echo "[Benchmark] Run HRNet MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-HRNetW18-Seg --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_num_threads=10
# UNet MKLDNN fp32
echo "[Benchmark] Run UNet MKLDNN int8"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-UNet --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_num_threads=10
# Deeplabv3-ResNet50 MKLDNN fp32
echo "[Benchmark] Run Deeplabv3-ResNet50 MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-Deeplabv3-ResNet50 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_num_threads=10
