export FLAGS_call_stack_level=2

# PPYOLOE MKLDNN
echo "[Benchmark] Run PPYOLOE MKLDNN"
python test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --precision=int8
# PicoDet MKLDNN
echo "[Benchmark] Run PicoDet MKLDNN"
python test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --precision=int8
# YOLOv5s MKLDNN
echo "[Benchmark] Run YOLOv5s MKLDNN"
python test_yolo_series_infer.py --model_path=models/yolov5s_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv5 --precision=int8
# YOLOv6s MKLDNN
echo "[Benchmark] Run YOLOv6s MKLDNN"
python test_yolo_series_infer.py --model_path=models/yolov6s_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv6 --precision=int8
# YOLOv7 MKLDNN
echo "[Benchmark] Run YOLOv7 MKLDNN"
python test_yolo_series_infer.py --model_path=models/yolov7_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv7 --precision=int8

# ResNet_vd MKLDNN
echo "[Benchmark] Run ResNet_vd MKLDNN"
python test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True --use_int8=True
# MobileNetV3_large MKLDNN
echo "[Benchmark] Run MobileNetV3_large MKLDNN"
python test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True --use_int8=True
# PPLCNetV2 MKLDNN
echo "[Benchmark] Run PPLCNetV2 MKLDNN"
python test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True --use_int8=True
# PPHGNet_tiny MKLDNN
echo "[Benchmark] Run PPHGNet_tiny MKLDNN"
python test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT --cpu_num_threads=10e --use_mkldnn=True --eval=True --use_int8=True
# EfficientNetB0 MKLDNN
echo "[Benchmark] Run EfficientNetB0 MKLDNN"
python test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT --cpu_num_threads=10 --use_mkldnn=True --eval=True --use_int8=True
