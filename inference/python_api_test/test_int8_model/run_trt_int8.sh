export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE trt int8
echo "[Benchmark] Run PPYOLOE trt int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --use_trt=True --precision=int8
# PicoDet trt int8
echo "[Benchmark] Run PicoDet trt int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --use_trt=True --precision=int8
# YOLOv5s trt int8
echo "[Benchmark] Run YOLOv5s trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_quant --use_trt=True --precision=int8 --arch=YOLOv5
# YOLOv6s trt int8
echo "[Benchmark] Run YOLOv6s trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_quant --use_trt=True --precision=int8 --arch=YOLOv6
# YOLOv7 trt int8
echo "[Benchmark] Run YOLOv7 trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_quant --use_trt=True --precision=int8 --arch=YOLOv7

# ResNet_vd trt int8
echo "[Benchmark] Run ResNet_vd trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT --use_trt=True --use_int8=True --eval=True --use_gpu=True
# MobileNetV3_large trt int8
echo "[Benchmark] Run MobileNetV3_large trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT --use_trt=True --use_int8=True --eval=True --use_gpu=True
# PPLCNetV2 trt int8
echo "[Benchmark] Run PPLCNetV2 trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT --use_trt=True --use_int8=True --eval=True --use_gpu=True
# PPHGNet_tiny trt int8
echo "[Benchmark] Run PPHGNet_tiny trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT --use_trt=True --use_int8=True --eval=True --use_gpu=True
# EfficientNetB0 trt int8
echo "[Benchmark] Run EfficientNetB0 trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT --use_trt=True --use_int8=True --eval=True --use_gpu=True

# PP-HumanSeg-Lite trt int8
echo "[Benchmark] Run PP-HumanSeg-Lite trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_humanseg_qat --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --use_trt=True --precision=int8
# PP-Liteseg trt int8
echo "[Benchmark] Run PP-Liteseg trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_liteseg_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=int8
# HRNet trt int8
echo "[Benchmark] Run HRNet trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/hrnet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=int8
# UNet trt int8
echo "[Benchmark] Run UNet trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/unet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=int8
# Deeplabv3-ResNet50 trt int8
echo "[Benchmark] Run Deeplabv3-ResNet50 trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/deeplabv3_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --use_trt=True --precision=int8

# ERNIE 3.0-Medium trt int8
echo "[Benchmark] Run ERNIE 3.0-Medium trt int8"
$PYTHON test_nlp_infer.py --model_path=models/save_ernie3_afqmc_new_cablib --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --use_trt --precision=int8
# PP-MiniLM MKLDNN trt int8
echo "[Benchmark] Run PP-MiniLM trt int8"
# BERT Base MKLDNN trt int8
echo "[Benchmark] Run BERT Base trt int8"
$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola_new_calib --use_trt --precision=int8 --batch_size=1
