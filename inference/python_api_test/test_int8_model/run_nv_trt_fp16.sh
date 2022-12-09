export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE+ trt fp16
echo "[Benchmark] Run PPYOLOE+ trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_plus_crn_s_80e_coco_no_nms/ppyoloe_plus_crn_s_80e_coco_no_nms.onnx --reader_config=configs/ppyoloe_plus_reader.yml --deploy_backend=tensorrt --precision=fp16 --model_name=PPYOLOE_PLUS --exclude_nms
# PicoDet trt fp16
echo "[Benchmark] Run PicoDet trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_no_postprocess/picodet_s_416_coco_npu_no_postprocess.onnx --reader_config=configs/picodet_reader.yml --deploy_backend=tensorrt --precision=fp16 --model_name=PicoDet --img_shape=416 --exclude_nms
# YOLOv5s trt fp16
echo "[Benchmark] Run YOLOv5s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_infer/yolov5s.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=YOLOv5s
# YOLOv6s trt fp16
echo "[Benchmark] Run YOLOv6s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_infer/yolov6s.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=YOLOv6s
# YOLOv7 trt fp16
echo "[Benchmark] Run YOLOv7 trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_infer/yolov7.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=YOLOv7


# ResNet_vd trt fp16
rm -rf model_fp16_model.trt
echo "[Benchmark] Run ResNet_vd trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_infer/model.onnx --deploy_backend=tensorrt --input_name=inputs --precision=fp16 --model_name=ResNet_vd
rm -rf model_fp16_model.trt
# MobileNetV3_large trt fp16
echo "[Benchmark] Run MobileNetV3_large trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_infer/model.onnx --deploy_backend=tensorrt --input_name=inputs --precision=fp16 --model_name=MobileNetV3_large
rm -rf model_fp16_model.trt
# PPLCNetV2 trt fp16
echo "[Benchmark] Run PPLCNetV2 trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_infer/model.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=PPLCNetV2
rm -rf model_fp16_model.trt
# PPHGNet_tiny trt fp16
echo "[Benchmark] Run PPHGNet_tiny trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_infer/model.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=PPHGNet_tiny
rm -rf model_fp16_model.trt
# EfficientNetB0 trt fp16
echo "[Benchmark] Run EfficientNetB0 trt fp16"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_infer/model.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=EfficientNetB0
rm -rf model_fp16_model.trt


# PP-HumanSeg-Lite trt fp16
echo "[Benchmark] Run PP-HumanSeg-Lite trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/pp_humanseg_fp32/pp_humanseg_fp32.onnx --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --deploy_backend=tensorrt --precision=fp16 --model_name=PP-HumanSeg-Lite
# PP-Liteseg trt fp16
echo "[Benchmark] Run PP-Liteseg trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/pp_liteseg_fp32/pp_liteseg_fp32.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=fp16 --model_name=ppliteseg
# HRNet trt fp16
echo "[Benchmark] Run HRNet trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/hrnet_fp32/hrnet_fp32.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=fp16 --model_name=HRNet
# UNet trt fp16
echo "[Benchmark] Run UNet trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/unet_fp32/unet_fp32.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=fp16 --model_name=UNet
# Deeplabv3-ResNet50 trt fp16
echo "[Benchmark] Run Deeplabv3-ResNet50 trt fp16"
$PYTHON test_segmentation_infer.py --model_path=models/Deeplabv3_ResNet50_fp32/Deeplabv3_ResNet50_fp32.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=fp16 --model_name=Deeplabv3-ResNet50


# nlp
# models/AFQMC
rm -rf model_fp16_model.trt
$PYTHON test_nlp_infer.py --model_path=models/AFQMC/model.onnx  --deploy_backend=tensorrt --task_name='afqmc' --precision=fp16 --model_name=ERNIE_3.0-Medium
# models/afqmc
$PYTHON test_nlp_infer.py --model_path=models/afqmc/model.onnx --deploy_backend=tensorrt  --task_name='afqmc' --precision=fp16 --model_name=PP-MiniLM
rm -rf model_fp16_model.trt
# models/x2paddle_cola
$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola/model.onnx--deploy_backend=tensorrt --precision=fp16 --batch_size=1 --model_name=BERT_Base
rm -rf model_fp16_model.trt
