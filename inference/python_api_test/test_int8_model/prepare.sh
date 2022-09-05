pip install -r requirements.txt

mkdir dataset
# download coco val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/coco_val2017.tar
tar -xf coco_val2017.tar -C ./dataset
rm -rf coco_val2017.tar
# download imagenet val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ILSVRC2012_val.tar
tar -xf ILSVRC2012_val.tar -C ./dataset
rm -rf ILSVRC2012_val.tar

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
# PicoDet
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu_quant.tar
tar -xf picodet_s_416_coco_npu_quant.tar -C ./models
rm -rf picodet_s_416_coco_npu_quant.tar
# Resnet50_vd
wget https://paddle-slim-models.bj.bcebos.com/act/ResNet50_vd_QAT.tar
tar -xf ResNet50_vd_QAT.tar -C ./models
rm -rf ResNet50_vd_QAT.tar
# MobileNetV3_large
wget https://paddle-slim-models.bj.bcebos.com/act/MobileNetV3_large_x1_0_QAT.tar
tar -xf MobileNetV3_large_x1_0_QAT.tar -C ./models
rm -rf MobileNetV3_large_x1_0_QAT.tar
# PPLCNetV2
wget https://paddle-slim-models.bj.bcebos.com/act/PPLCNetV2_base_QAT.tar
tar -xf PPLCNetV2_base_QAT.tar -C ./models
rm -rf PPLCNetV2_base_QAT.tar
# PPHGNet_tiny
wget https://paddle-slim-models.bj.bcebos.com/act/PPHGNet_tiny_QAT.tar
tar -xf PPHGNet_tiny_QAT.tar -C ./models
rm -rf PPHGNet_tiny_QAT.tar
# EfficientNetB0
wget https://paddle-slim-models.bj.bcebos.com/act/EfficientNetB0_QAT.tar
tar -xf EfficientNetB0_QAT.tar -C ./models
rm -rf EfficientNetB0_QAT.tar


# ====== download FP32 inference model ======
# PPYOLOE
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar
tar -xf ppyoloe_crn_l_300e_coco.tar -C ./models
rm -rf ppyoloe_crn_l_300e_coco.tar
# YOLOv5s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_infer.tar
tar -xf yolov5s_infer.tar -C ./models
rm -rf yolov5s_infer.tar
# YOLOv6s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_infer.tar
tar -xf yolov6s_infer.tar -C ./models
rm -rf yolov6s_infer.tar
# YOLOv7
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_infer.tar
tar -xf yolov7_infer.tar -C ./models
rm -rf yolov7_infer.tar
# PicoDet
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar
tar -xf picodet_s_416_coco_npu.tar -C ./models
rm -rf picodet_s_416_coco_npu.tar
# Resnet50_vd
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
tar -xf ResNet50_vd_infer.tar -C ./models
rm -rf ResNet50_vd_infer.tar
# MobileNetV3_large
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
tar -xf MobileNetV3_large_x1_0_infer.tar -C ./models
rm -rf MobileNetV3_large_x1_0_infer.tar
# PPLCNetV2
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNetV2_base_infer.tar
tar -xf PPLCNetV2_base_infer.tar -C ./models
rm -rf PPLCNetV2_base_infer.tar
# PPHGNet_tiny
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_tiny_infer.tar
tar -xf PPHGNet_tiny_infer.tar -C ./models
rm -rf PPHGNet_tiny_infer.tar
# EfficientNetB0
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/EfficientNetB0_infer.tar
tar -xf EfficientNetB0_infer.tar -C ./models
rm -rf EfficientNetB0_infer.tar
