export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE+ trt int8
echo "[Benchmark] Run PPYOLOE+ trt int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant/ppyoloe_plus_crn_s_80e_coco_no_nms_quant.onnx --reader_config=configs/ppyoloe_plus_reader.yml --deploy_backend=tensorrt --precision=int8 --model_name=PPYOLOE_PLUS --calibration_file=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant/calibration.cache --exclude_nms
# PicoDet trt int8
echo "[Benchmark] Run PicoDet trt int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_no_postprocess_quant/picodet_s_416_coco_npu_no_postprocess_quant.onnx --reader_config=configs/picodet_reader.yml --deploy_backend=tensorrt --precision=int8 --model_name=PicoDet --calibration_file=models/picodet_s_416_coco_npu_no_postprocess_quant/calibration.cache --img_shape=416 --exclude_nms
# YOLOv5s trt int8
echo "[Benchmark] Run YOLOv5s trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_quant/yolov5s_quant.onnx --deploy_backend=tensorrt --precision=int8 --model_name=YOLOv5s --calibration_file=models/yolov5s_quant/calibration.cache
# YOLOv6s trt int8
echo "[Benchmark] Run YOLOv6s trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_quant/yolov6s_quant.onnx --deploy_backend=tensorrt --precision=int8 --model_name=YOLOv6s --calibration_file=models/yolov6s_quant/calibration.cache
# YOLOv7 trt int8
echo "[Benchmark] Run YOLOv7 trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_quant/yolov7_quant.onnx --deploy_backend=tensorrt --precision=int8 --model_name=YOLOv7 --calibration_file=models/yolov7_quant/calibration.cache

# ResNet_vd trt int8
rm -rf model_int8_model.trt
echo "[Benchmark] Run ResNet_vd trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT/model.onnx --deploy_backend=tensorrt --input_name=inputs --precision=int8 --calibration_file=models/ResNet50_vd_QAT/calibration.cache --model_name=ResNet_vd
rm -rf model_int8_model.trt
# MobileNetV3_large trt int8
echo "[Benchmark] Run MobileNetV3_large trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT/model.onnx --deploy_backend=tensorrt --input_name=inputs --precision=int8 --calibration_file=models/MobileNetV3_large_x1_0_QAT/calibration.cache --model_name=MobileNetV3_large
rm -rf model_int8_model.trt
# PPLCNetV2 trt int8
echo "[Benchmark] Run PPLCNetV2 trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT/model.onnx --deploy_backend=tensorrt --precision=int8 --calibration_file=models/PPLCNetV2_base_QAT/calibration.cache --model_name=PPLCNetV2
rm -rf model_int8_model.trt
# PPHGNet_tiny trt int8
echo "[Benchmark] Run PPHGNet_tiny trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT/model.onnx --deploy_backend=tensorrt --precision=int8 --calibration_file=models/PPHGNet_tiny_QAT/calibration.cache  --model_name=PPHGNet_tiny
rm -rf model_int8_model.trt
# EfficientNetB0 trt int8
echo "[Benchmark] Run EfficientNetB0 trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT/model.onnx --deploy_backend=tensorrt --precision=int8 --calibration_file=models/EfficientNetB0_QAT/calibration.cache --model_name=EfficientNetB0
rm -rf model_int8_model.trt

# PP-HumanSeg-Lite trt int8
echo "[Benchmark] Run PP-HumanSeg-Lite trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_humanseg_int8/pp_humanseg_int8.onnx --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --deploy_backend=tensorrt --precision=int8 --model_name=PP-HumanSeg-Lite --calibration_file=models/pp_humanseg_int8/calibration.cache
# PP-Liteseg trt int8
echo "[Benchmark] Run PP-Liteseg trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_liteseg_int8/pp_liteseg_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=PP-Liteseg --calibration_file=models/pp_liteseg_int8/calibration.cache
# HRNet trt int8
echo "[Benchmark] Run HRNet trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/hrnet_int8/hrnet_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=HRNet --calibration_file=models/hrnet_int8/calibration.cache
# UNet trt int8
echo "[Benchmark] Run UNet trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/unet_int8/unet_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=UNet --calibration_file=models/unet_int8/calibration.cache
# Deeplabv3-ResNet50 trt int8
echo "[Benchmark] Run Deeplabv3-ResNet50 trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/deeplabv3_int8/deeplabv3_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=Deeplabv3-ResNet50 --calibration_file=models/deeplabv3_int8/calibration.cache

# ERNIE 3.0-Medium nv-trt int8
rm -rf model_int8_model.trt
echo "[Benchmark] Run NV-TRT ERNIE 3.0-Medium trt int8"
$PYTHON  test_nlp_infer.py --model_path=models/save_ernie3_afqmc_new_cablib/model.onnx --deploy_backend=tensorrt  --task_name='afqmc' --precision=int8 --calibration_file=models/save_ernie3_afqmc_new_cablib/calibration.cache --model_name=ERNIE_3.0-Medium
rm -rf model_int8_model.trt
# PP-MiniLM nv-trt int8
echo "[Benchmark] Run NV-TRT PP-MiniLM trt int8"
$PYTHON test_nlp_infer.py --model_path=models/save_ppminilm_afqmc_new_calib/model.onnx --deploy_backend=tensorrt  --task_name='afqmc' --precision=int8 --calibration_file=models/save_ppminilm_afqmc_new_calib/calibration.cache --model_name=PP-MiniLM
rm -rf model_int8_model.trt
# BERT Base nv-trt int8
echo "[Benchmark] Run NV-TRT BERT Base trt int8"
$PYTHON  test_bert_infer.py --model_path=models/x2paddle_cola_new_calib/model.onnx --precision=int8 --batch_size=1 --deploy_backend=tensorrt  --calibration_file=./models/x2paddle_cola_new_calib/calibration.cache --model_name=BERT_Base
rm -rf model_int8_model.trt
