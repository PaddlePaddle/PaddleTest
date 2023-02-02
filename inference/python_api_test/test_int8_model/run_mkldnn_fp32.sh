export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE MKLDNN
echo "[Benchmark] Run PPYOLOE MKLDNN fp32"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco --reader_config=configs/ppyoloe_reader.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=PPYOLOE
# PicoDet MKLDNN
echo "[Benchmark] Run PicoDet MKLDNN fp32"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu --reader_config=configs/picodet_reader.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=PicoDet
# YOLOv5s MKLDNN
echo "[Benchmark] Run YOLOv5s MKLDNN fp32"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_infer --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=YOLOv5s
# YOLOv6s MKLDNN
echo "[Benchmark] Run YOLOv6s MKLDNN fp32"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_infer --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=YOLOv6s
# YOLOv7 MKLDNN
echo "[Benchmark] Run YOLOv7 MKLDNN fp32"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_infer --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=YOLOv7

# ResNet_vd MKLDNN
echo "[Benchmark] Run ResNet_vd MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_infer --cpu_num_threads=10 --use_mkldnn=True --model_name=ResNet_vd
# MobileNetV3_large MKLDNN
echo "[Benchmark] Run MobileNetV3_large MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_infer --cpu_num_threads=10 --use_mkldnn=True --model_name=MobileNetV3_large
# PPLCNetV2 MKLDNN
echo "[Benchmark] Run PPLCNetV2 MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_infer --cpu_num_threads=10 --use_mkldnn=True --model_name=PPLCNetV2
# PPHGNet_tiny MKLDNN
echo "[Benchmark] Run PPHGNet_tiny MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_infer --cpu_num_threads=10 --use_mkldnn=True --model_name=PPHGNet_tiny
# EfficientNetB0 MKLDNN
echo "[Benchmark] Run EfficientNetB0 MKLDNN fp32"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_infer --cpu_num_threads=10 --use_mkldnn=True --model_name=EfficientNetB0

# PP-HumanSeg-Lite MKLDNN fp32
echo "[Benchmark] Run PP-HumanSeg-Lite MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/ppseg_lite_portrait_398x224_with_softmax --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=PP-HumanSeg-Lite
# PP-Liteseg MKLDNN fp32
echo "[Benchmark] Run PP-Liteseg MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-PPLIteSegSTDC1 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=PP-Liteseg
# HRNet MKLDNN fp32
echo "[Benchmark] Run HRNet MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-HRNetW18-Seg --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=HRNet
# UNet MKLDNN fp32
echo "[Benchmark] Run UNet MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-UNet --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=UNet
# Deeplabv3-ResNet50 MKLDNN fp32
echo "[Benchmark] Run Deeplabv3-ResNet50 MKLDNN fp32"
$PYTHON test_segmentation_infer.py --model_path=models/RES-paddle2-Deeplabv3-ResNet50 --model_filename=model --params_filename=params --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=Deeplabv3-ResNet50

# ERNIE 3.0-Medium MKLDNN fp32
echo "[Benchmark] Run ERNIE 3.0-Medium MKLDNN fp32"
$PYTHON test_nlp_infer.py --model_path=models/AFQMC --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=ERNIE_3.0-Medium
# PP-MiniLM MKLDNN fp32
echo "[Benchmark] Run PP-MiniLM MKLDNN fp32"
$PYTHON test_nlp_infer.py --model_path=models/afqmc --task_name='afqmc' --device=CPU --use_mkldnn=True --cpu_threads=10 --model_name=PP-MiniLM
# BERT Base MKLDNN fp32
echo "[Benchmark] Run BERT Base MKLDNN fp32"
$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola --device=CPU --use_mkldnn=True --cpu_threads=10 --batch_size=1 --model_name=BERT_Base
