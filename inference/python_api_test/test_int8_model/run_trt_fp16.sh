export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2

# PPYOLOE trt fp16
echo "[Benchmark] Run PPYOLOE trt fp16"
python test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco --reader_config=configs/ppyoloe_reader.yml --use_trt=True --precision=fp16
# PicoDet trt fp16
echo "[Benchmark] Run PicoDet trt fp16"
python test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --use_trt=True --precision=fp16
# YOLOv5s trt fp16
echo "[Benchmark] Run YOLOv5s trt fp16"
python test_yolo_series_infer.py --model_path=models/yolov5s_infer --use_trt=True --precision=fp16 --arch=YOLOv5
# YOLOv6s trt fp16
echo "[Benchmark] Run YOLOv6s trt fp16"
python test_yolo_series_infer.py --model_path=models/yolov6s_infer --use_trt=True --precision=fp16 --arch=YOLOv6
# YOLOv7 trt fp16
echo "[Benchmark] Run YOLOv7 trt fp16"
python test_yolo_series_infer.py --model_path=models/yolov7_infer --use_trt=True --precision=fp16 --arch=YOLOv7

# ResNet_vd trt fp16
echo "[Benchmark] Run ResNet_vd trt fp16"
python test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT --use_trt=True --use_fp16=True --eval=True --use_gpu=True
# MobileNetV3_large trt fp16
echo "[Benchmark] Run MobileNetV3_large trt fp16"
python test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT --use_trt=True --use_fp16=True --eval=True --use_gpu=True
# PPLCNetV2 trt fp16
echo "[Benchmark] Run PPLCNetV2 trt fp16"
python test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT --use_trt=True --use_fp16=True --eval=True --use_gpu=True
# PPHGNet_tiny trt fp16
echo "[Benchmark] Run PPHGNet_tiny trt fp16"
python test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT --use_trt=True --use_fp16=True --eval=True --use_gpu=True
# EfficientNetB0 trt fp16
echo "[Benchmark] Run EfficientNetB0 trt fp16"
python test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT --use_trt=True --use_fp16=True --eval=True --use_gpu=True
