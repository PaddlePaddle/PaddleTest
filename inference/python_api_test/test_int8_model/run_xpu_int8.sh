#export XPU_VISIBLE_DEVICES=2
export FLAGS_call_stack_level=2
# Add this to reduce cpu memory!
export CUDA_MODULE_LOADING=LAZY
PYTHON="python"

# PPYOLOE XPU int8
echo "[Benchmark] Run PPYOLOE XPU int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --precision=int8 --model_name=PPYOLOE --device=XPU
#$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --precision=int8 --model_name=PPYOLOE --device=XPU --use_l3=True
# PPYOLOE+ XPU int8
echo "[Benchmark] Run PPYOLOE+ XPU int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant --reader_config=configs/ppyoloe_plus_reader.yml --precision=int8 --model_name=PPYOLOE_PLUS --exclude_nms --device=XPU
#$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant --reader_config=configs/ppyoloe_plus_reader.yml --precision=int8 --model_name=PPYOLOE_PLUS --exclude_nms --device=XPU --use_l3=True
# PicoDet XPU int8
### echo "[Benchmark] Run PicoDet XPU int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --precision=int8 --model_name=PicoDet --device=XPU
#$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --precision=int8 --model_name=PicoDet --device=XPU --use_l3=True
# PicoDet no nms XPU int8
echo "[Benchmark] Run PicoDet no nms XPU int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_no_postprocess_quant --reader_config=configs/picodet_reader.yml --precision=int8 --model_name=PicoDet_no_postprocess --exclude_nms --device=XPU
#$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_no_postprocess_quant --reader_config=configs/picodet_reader.yml --precision=int8 --model_name=PicoDet_no_postprocess --exclude_nms --device=XPU --use_l3=True
# YOLOv5s XPU int8
echo "[Benchmark] Run YOLOv5s XPU int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_quant --precision=int8 --model_name=YOLOv5s --device=XPU
#$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_quant --precision=int8 --model_name=YOLOv5s --device=XPU --use_l3=True
# YOLOv6s XPU int8
echo "[Benchmark] Run YOLOv6s XPU int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_quant --precision=int8 --model_name=YOLOv6s --device=XPU
#$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_quant --precision=int8 --model_name=YOLOv6s --device=XPU --use_l3=True
# YOLOv7 XPU int8
echo "[Benchmark] Run YOLOv7 XPU int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_quant --precision=int8 --model_name=YOLOv7 --device=XPU
#$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_quant --precision=int8 --model_name=YOLOv7 --device=XPU --use_l3=True

# ResNet_vd XPU int8
echo "[Benchmark] Run ResNet_vd XPU int8"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT --precision=int8 --model_name=ResNet_vd --device=XPU
#$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT --precision=int8 --model_name=ResNet_vd --device=XPU --use_l3=True
# MobileNetV3_large XPU int8
echo "[Benchmark] Run MobileNetV3_large XPU int8"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT --precision=int8 --model_name=MobileNetV3_large --device=XPU
#$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT --precision=int8 --model_name=MobileNetV3_large --device=XPU --use_l3=True
# PPLCNetV2 XPU int8
echo "[Benchmark] Run PPLCNetV2 XPU int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT --precision=int8 --model_name=PPLCNetV2 --device=XPU
#$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT --precision=int8 --model_name=PPLCNetV2 --device=XPU --use_l3=True
# PPHGNet_tiny XPU int8
echo "[Benchmark] Run PPHGNet_tiny XPU int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT --precision=int8 --model_name=PPHGNet_tiny --device=XPU
#$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT --precision=int8 --model_name=PPHGNet_tiny --device=XPU --use_l3=True
# EfficientNetB0 XPU int8
echo "[Benchmark] Run EfficientNetB0 XPU int8"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT --precision=int8 --model_name=EfficientNetB0 --device=XPU
#$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT --precision=int8 --model_name=EfficientNetB0 --device=XPU --use_l3=True

# PP-HumanSeg-Lite XPU int8
#echo "[Benchmark] Run PP-HumanSeg-Lite XPU int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_humanseg_qat --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --precision=int8 --model_name=PP-HumanSeg-Lite --device=XPU
#$PYTHON test_segmentation_infer.py --model_path=models/pp_humanseg_qat --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --precision=int8 --model_name=PP-HumanSeg-Lite --device=XPU --use_l3=True
# PP-Liteseg XPU int8
echo "[Benchmark] Run PP-Liteseg XPU int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_liteseg_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=PP-Liteseg --device=XPU
#$PYTHON test_segmentation_infer.py --model_path=models/pp_liteseg_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=PP-Liteseg --device=XPU --use_l3=True
# HRNet XPU int8
echo "[Benchmark] Run HRNet XPU int8"
$PYTHON test_segmentation_infer.py --model_path=models/hrnet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=HRNet --device=XPU
#$PYTHON test_segmentation_infer.py --model_path=models/hrnet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=HRNet --device=XPU --use_l3=True
# UNet XPU int8
echo "[Benchmark] Run UNet XPU int8"
$PYTHON test_segmentation_infer.py --model_path=models/unet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=UNet --device=XPU
#$PYTHON test_segmentation_infer.py --model_path=models/unet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=UNet --device=XPU --use_l3=True
# Deeplabv3-ResNet50 XPU int8
echo "[Benchmark] Run Deeplabv3-ResNet50 XPU int8"
$PYTHON test_segmentation_infer.py --model_path=models/deeplabv3_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=Deeplabv3-ResNet50 --device=XPU
#$PYTHON test_segmentation_infer.py --model_path=models/deeplabv3_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=int8 --model_name=Deeplabv3-ResNet50 --device=XPU --use_l3=True

# ERNIE 3.0-Medium XPU int8 该模型不支持int8精度，暂时关闭
echo "[Benchmark] Run ERNIE 3.0-Medium XPU int8"
#$PYTHON test_nlp_infer.py --model_path=models/save_ernie3_afqmc_new_cablib --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --precision=int8 --model_name=ERNIE_3.0-Medium --device=XPU
#$PYTHON test_nlp_infer.py --model_path=models/save_ernie3_afqmc_new_cablib --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --precision=int8 --model_name=ERNIE_3.0-Medium --device=XPU --use_l3=True
# PP-MiniLM MKLDNN XPU int8
echo "[Benchmark] Run PP-MiniLM XPU int8"
$PYTHON test_nlp_infer.py --model_path=models/save_ppminilm_afqmc_new_calib --task_name='afqmc' --precision=int8 --model_name=PP-MiniLM --device=XPU
#$PYTHON test_nlp_infer.py --model_path=models/save_ppminilm_afqmc_new_calib --task_name='afqmc' --precision=int8 --model_name=PP-MiniLM --device=XPU --use_l3=True
# BERT Base MKLDNN XPU int8
echo "[Benchmark] Run BERT Base XPU int8"
$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola_new_calib --precision=int8 --batch_size=1 --model_name=BERT_Base --device=XPU
#$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola_new_calib --precision=int8 --batch_size=1 --model_name=BERT_Base --device=XPU --use_l3=True
