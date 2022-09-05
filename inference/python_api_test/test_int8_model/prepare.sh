pip install requirements.txt

mkdir dataset
# download coco val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/coco_val2017.tar
tar -xf coco_val2017.tar -C ./dataset
rm -rf coco_val2017.tar

# ====== download inference model ======
# PPYOLOE
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar
tar -xf ppyoloe_crn_l_300e_coco_quant.tar
rm -rf ppyoloe_crn_l_300e_coco_quant.tar
# YOLOv5s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar
tar -xf yolov5s_quant.tar
rm -rf yolov5s_quant.tar
# YOLOv5s
wget hhttps://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant.tar
tar -xf yolov6s_quant.tar
rm -rf yolov6s_quant.tar
# YOLOv7
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant.tar
tar -xf yolov7_quant.tar
rm -rf yolov7_quant.tar
