pip install -r requirements.txt

# ======== download Val Dataset
mkdir dataset
# download coco val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/coco_val2017.tar
tar -xf coco_val2017.tar -C ./dataset
rm -rf coco_val2017.tar
# download imagenet val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ILSVRC2012_val.tar
tar -xf ILSVRC2012_val.tar -C ./dataset
rm -rf ILSVRC2012_val.tar
# download cityscapes val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/cityscapes_val.tar
tar -xf cityscapes_val.tar -C ./dataset
rm -rf cityscapes_val.tar
# download portrait14k val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/portrait14k_val.tar
tar -xf portrait14k_val.tar -C ./dataset
rm -rf portrait14k_val.tar

mkdir models

# ====== download INT8 quant inference model ======
# PPYOLOE with nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar
tar -xf ppyoloe_crn_l_300e_coco_quant.tar -C ./models
rm -rf ppyoloe_crn_l_300e_coco_quant.tar
# PPYOLOE+ without nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_plus_crn_s_80e_coco_no_nms_quant.tar
tar -xf ppyoloe_plus_crn_s_80e_coco_no_nms_quant.tar -C ./models
rm -rf ppyoloe_plus_crn_s_80e_coco_no_nms_quant.tar
# PicoDet with nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu_quant.tar
tar -xf picodet_s_416_coco_npu_quant.tar -C ./models
rm -rf picodet_s_416_coco_npu_quant.tar
# PicoDet without postprocess
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu_no_postprocess_quant.tar
tar -xf picodet_s_416_coco_npu_no_postprocess_quant.tar -C ./models
rm -rf picodet_s_416_coco_npu_no_postprocess_quant.tar
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
# PP-HumanSeg-Lite
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/pp_humanseg_qat.tar
tar -xf pp_humanseg_qat.tar -C ./models
rm -rf pp_humanseg_qat.tar
# PP-Liteseg
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/pp_liteseg_qat.tar
tar -xf pp_liteseg_qat.tar -C ./models
rm -rf pp_liteseg_qat.tar
# HRNet
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/hrnet_qat.tar
tar -xf hrnet_qat.tar -C ./models
rm -rf hrnet_qat.tar
# UNet
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/unet_qat.tar
tar -xf unet_qat.tar -C ./models
rm -rf unet_qat.tar
# Deeplabv3-ResNet50
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/deeplabv3_qat.tar
tar -xf deeplabv3_qat.tar -C ./models
rm -rf deeplabv3_qat.tar
# ERNIE 3.0-Medium
wget https://bj.bcebos.com/v1/paddle-slim-models/act/save_ernie3_afqmc_new_cablib.tar
tar -xf save_ernie3_afqmc_new_cablib.tar -C ./models
rm -rf save_ernie3_afqmc_new_cablib.tar
# PP-MiniLM
wget https://bj.bcebos.com/v1/paddle-slim-models/act/save_ppminilm_afqmc_new_calib.tar
tar -xf save_ppminilm_afqmc_new_calib.tar -C ./models
rm -rf save_ppminilm_afqmc_new_calib.tar
# BERT Base
wget https://bj.bcebos.com/v1/paddle-slim-models/act/x2paddle_cola_new_calib.tar
tar -xf x2paddle_cola_new_calib.tar -C ./models
rm -rf x2paddle_cola_new_calib.tar

# ====== download FP32 inference model ======
# PPYOLOE with nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar
tar -xf ppyoloe_crn_l_300e_coco.tar -C ./models
rm -rf ppyoloe_crn_l_300e_coco.tar
# PPYOLOE+ without nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_plus_crn_s_80e_coco_no_nms.tar
tar -xf ppyoloe_plus_crn_s_80e_coco_no_nms.tar -C ./models
rm -rf ppyoloe_plus_crn_s_80e_coco_no_nms.tar
# PicoDet with nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar
tar -xf picodet_s_416_coco_npu.tar -C ./models
rm -rf picodet_s_416_coco_npu.tar
# PicoDet without postprocess
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu_no_postprocess.tar
tar -xf picodet_s_416_coco_npu_no_postprocess.tar -C ./models
rm -rf picodet_s_416_coco_npu_no_postprocess.tar
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
# PP-HumanSeg-Lite
wget https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz
tar -xf ppseg_lite_portrait_398x224_with_softmax.tar.gz -C ./models
rm -rf ppseg_lite_portrait_398x224_with_softmax.tar.gz
# PP-Liteseg
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-PPLIteSegSTDC1.zip
unzip -q RES-paddle2-PPLIteSegSTDC1.zip -d ./models
rm -rf RES-paddle2-PPLIteSegSTDC1.zip
# HRNet
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-HRNetW18-Seg.zip
unzip -q RES-paddle2-HRNetW18-Seg.zip -d ./models
rm -rf RES-paddle2-HRNetW18-Seg.zip
# UNet
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-UNet.zip
unzip -q RES-paddle2-UNet.zip -d ./models
rm -rf RES-paddle2-UNet.zip
# Deeplabv3-ResNet50
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-Deeplabv3-ResNet50.zip
unzip -q RES-paddle2-Deeplabv3-ResNet50.zip -d ./models
rm -rf RES-paddle2-Deeplabv3-ResNet50.zip
# ERNIE 3.0-Medium
wget https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/AFQMC.tar
tar -xf AFQMC.tar -C ./models
rm -rf AFQMC.tar
# PP-MiniLM
wget https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar
tar -xf afqmc.tar -C ./models
rm -rf afqmc.tar
# BERT Base
wget https://paddle-slim-models.bj.bcebos.com/act/x2paddle_cola.tar
tar xf x2paddle_cola.tar -C ./models
rm -rf x2paddle_cola.tar


# ====== ocr model and datset======
# download val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr_det/test_set.tar
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr_det/test_label.txt
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr_det/test.jpg
tar -xf test_set.tar

# download inference model
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar # fp32
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PPOCRV3_det_QAT.tar
tar -xvf PPOCRV3_det_QAT.tar # int8

git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
