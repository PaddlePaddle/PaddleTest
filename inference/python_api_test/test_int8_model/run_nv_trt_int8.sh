export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE trt int8
echo "[Benchmark] Run PPYOLOE trt int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant/ppyoloe_crn_l_300e_coco_quant.onnx --reader_config=configs/ppyoloe_reader.yml --deploy_backend=tensorrt --precision=int8 --model_name=PPYOLOE --calibration_file=models/ppyoloe_crn_l_300e_coco_quant/calibration.cache
# PicoDet trt int8
echo "[Benchmark] Run PicoDet trt int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant/picodet_s_416_coco_npu_quant.onnx --reader_config=configs/picodet_reader.yml --deploy_backend=tensorrt --precision=int8 --model_name=PicoDet --calibration_file=models/picodet_s_416_coco_npu_quant/calibration.cache
# YOLOv5s trt int8
echo "[Benchmark] Run YOLOv5s trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_quant/yolov5s_quant.onnx --deploy_backend=tensorrt --precision=int8 --model_name=YOLOv5s --calibration_file=models/yolov5s_quant/calibration.cache
# YOLOv6s trt int8
echo "[Benchmark] Run YOLOv6s trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_quant/yolov6s_quant.onnx --deploy_backend=tensorrt --precision=int8 --model_name=YOLOv6s --calibration_file=models/yolov6s_quant/calibration.cache
# YOLOv7 trt int8
echo "[Benchmark] Run YOLOv7 trt int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_quant/yolov7_quant.onnx --deploy_backend=tensorrt --precision=int8 --model_name=YOLOv7 --calibration_file=models/yolov7_quant/calibration.cache

# PP-HumanSeg-Lite trt int8
echo "[Benchmark] Run PP-HumanSeg-Lite trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_humanseg_int8/pp_humanseg_int8.onnx --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --deploy_backend=tensorrt --precision=int8 --model_name=PP-HumanSeg-Lite --calibration_file=models/pp_humanseg_int8/calibration.cache
# PP-Liteseg trt int8
echo "[Benchmark] Run PP-Liteseg trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_liteseg_int8/pp_liteseg_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=ppliteseg --calibration_file=models/pp_liteseg_int8/calibration.cache
# HRNet trt int8
echo "[Benchmark] Run HRNet trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/hrnet_int8/hrnet_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=HRNet --calibration_file=models/hrnet_int8/calibration.cache
# UNet trt int8
echo "[Benchmark] Run UNet trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/unet_int8/unet_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=UNet --calibration_file=models/unet_int8/calibration.cache
# Deeplabv3-ResNet50 trt int8
echo "[Benchmark] Run Deeplabv3-ResNet50 trt int8"
$PYTHON test_segmentation_infer.py --model_path=models/deeplabv3_int8/deeplabv3_int8.onnx --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --deploy_backend=tensorrt --precision=int8 --model_name=Deeplabv3-ResNet50 --calibration_file=models/deeplabv3_int8/calibration.cache
