pip install requirements.txt

mkdir dataset
# download coco val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/coco_val2017.tar
tar -xf coco_val2017.tar -C ./dataset
rm -rf coco_val2017.tar

mkdir models

# ====== download INT8 quant inference model ======
# PPYOLOE
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar
tar -xf ppyoloe_crn_l_300e_coco_quant.tar -C ./models
rm -rf ppyoloe_crn_l_300e_coco_quant.tar
# YOLOv5s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar
tar -xf yolov5s_quant.tar -C ./models
rm -rf yolov5s_quant.tar
# YOLOv5s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant.tar
tar -xf yolov6s_quant.tar -C ./models
rm -rf yolov6s_quant.tar
# YOLOv7
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant.tar
tar -xf yolov7_quant.tar -C ./models
rm -rf yolov7_quant.tar

# ====== download FP32 inference model ======
# PPYOLOE
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar
tar -xf ppyoloe_crn_l_300e_coco.tar -C ./models
rm -rf ppyoloe_crn_l_300e_coco.tar
# YOLOv5s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_infer.tar
tar -xf yolov5s_infer.tar -C ./models
rm -rf yolov5s_infer.tar
# YOLOv5s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_infer.tar
tar -xf yolov6s_infer.tar -C ./models
rm -rf yolov6s_infer.tar
# YOLOv7
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_infer.tar
tar -xf yolov7_infer.tar -C ./models
rm -rf yolov7_infer.tar
