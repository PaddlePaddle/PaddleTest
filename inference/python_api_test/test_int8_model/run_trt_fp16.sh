export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2

# PPYOLOE trt int8
python test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco --reader_config=configs/ppyoloe_reader.yml --run_mode=trt_fp16
# YOLOv5s trt int8
python test_yolo_series_infer.py --model_path=models/yolov5s_infer --run_mode=trt_fp16 --arch=YOLOv5
# YOLOv6s trt int8
python test_yolo_series_infer.py --model_path=models/yolov6s_infer --run_mode=trt_fp16 --arch=YOLOv6
# YOLOv7 trt int8
python test_yolo_series_infer.py --model_path=models/yolov7_infer --run_mode=trt_fp16 --arch=YOLOv7
