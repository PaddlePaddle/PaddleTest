export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2

# PPYOLOE trt int8
python test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --run_mode=trt_int8
# YOLOv5s trt int8
python test_yolo_series_infer.py --model_path=models/yolov5s_quant --run_mode=trt_int8 --arch=YOLOv5
# YOLOv6s trt int8
python test_yolo_series_infer.py --model_path=models/yolov6s_quant --run_mode=trt_int8 --arch=YOLOv6
# YOLOv7 trt int8
python test_yolo_series_infer.py --model_path=models/yolov7_quant --run_mode=trt_int8 --arch=YOLOv7
