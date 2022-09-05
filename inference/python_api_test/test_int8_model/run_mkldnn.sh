export FLAGS_call_stack_level=2

# PPYOLOE trt int8
python test_ppyoloe_infer.py --model_path=ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --device=CPU --enable_mkldnn=True --cpu_threads=10
# YOLOv5s trt int8
python test_yolo_series_infer.py --model_path=yolov5s_quant --device=CPU --enable_mkldnn=True --cpu_threads=10 --arch=YOLOv5
# YOLOv6s trt int8
python test_yolo_series_infer.py --model_path=yolov6s_quant --device=CPU --enable_mkldnn=True --cpu_threads=10 --arch=YOLOv6
# YOLOv7 trt int8
python test_yolo_series_infer.py --model_path=yolov7_quant --device=CPU --enable_mkldnn=True --cpu_threads=10 --arch=YOLOv7
