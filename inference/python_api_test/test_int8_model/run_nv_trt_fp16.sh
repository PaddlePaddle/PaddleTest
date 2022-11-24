export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE trt fp16
echo "[Benchmark] Run PPYOLOE trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --reader_config=configs/ppyoloe_reader.yml --deploy_backend=tensorrt --precision=fp16 --model_name=PPYOLOE
# PicoDet trt fp16
echo "[Benchmark] Run PicoDet trt fp16"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu/picodet_s_416_coco_npu.onnx --reader_config=configs/picodet_reader.yml --deploy_backend=tensorrt --precision=fp16 --model_name=PicoDet
# YOLOv5s trt fp16
echo "[Benchmark] Run YOLOv5s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_infer/yolov5s.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=YOLOv5s
# YOLOv6s trt fp16
echo "[Benchmark] Run YOLOv6s trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_infer/yolov6s.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=YOLOv6s
# YOLOv7 trt fp16
echo "[Benchmark] Run YOLOv7 trt fp16"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_infer/yolov7.onnx --deploy_backend=tensorrt --precision=fp16 --model_name=YOLOv7

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
