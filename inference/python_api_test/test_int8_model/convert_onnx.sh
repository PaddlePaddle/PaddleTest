python -m pip install paddle2onnx==1.0.3

# ================================ FP32 ======================================
# PPYOLOE+ no nms
paddle2onnx --model_dir=models/ppyoloe_plus_crn_s_80e_coco_no_nms/ --save_file=models/ppyoloe_plus_crn_s_80e_coco_no_nms/ppyoloe_plus_crn_s_80e_coco_no_nms.onnx --model_filename=model.pdmodel --params_filename=model.pdiparams
# PicoDet no nms
paddle2onnx --model_dir=models/picodet_s_416_coco_npu_no_postprocess/ --save_file=models/picodet_s_416_coco_npu_no_postprocess/picodet_s_416_coco_npu_no_postprocess.onnx --model_filename=model.pdmodel --params_filename=model.pdiparams
# YOLOv5s
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx
mv yolov5s.onnx models/yolov5s_infer
# YOLOv6s
wget https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx
mv yolov6s.onnx models/yolov6s_infer
# YOLOv7
wget https://paddle-slim-models.bj.bcebos.com/act/yolov7.onnx
mv yolov7.onnx models/yolov7_infer
# PP-HumanSeg-Lite
python utils/paddle_infer_shape.py --model_dir=models/ppseg_lite_portrait_398x224_with_softmax/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_dir=models/pp_humanseg_fp32 --input_shape_dict="{'x':[1, 3, 398, 224]}"
paddle2onnx --model_dir=models/pp_humanseg_fp32 --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/pp_humanseg_fp32/pp_humanseg_fp32.onnx
# PP-Liteseg
python utils/paddle_infer_shape.py --model_dir=models/RES-paddle2-PPLIteSegSTDC1/ --model_filename=model --params_filename=params --save_dir=models/pp_liteseg_fp32 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/pp_liteseg_fp32 --model_filename=model --params_filename=params --save_file=models/pp_liteseg_fp32/pp_liteseg_fp32.onnx
# HRNet
python utils/paddle_infer_shape.py --model_dir=models/RES-paddle2-HRNetW18-Seg/ --model_filename=model --params_filename=params --save_dir=models/hrnet_fp32 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/hrnet_fp32 --model_filename=model --params_filename=params --save_file=models/hrnet_fp32/hrnet_fp32.onnx
# UNet
python utils/paddle_infer_shape.py --model_dir=models/RES-paddle2-UNet/ --model_filename=model --params_filename=params --save_dir=models/unet_fp32 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/unet_fp32 --model_filename=model --params_filename=params --save_file=models/unet_fp32/unet_fp32.onnx
# Deeplabv3-ResNet50
python utils/paddle_infer_shape.py --model_dir=models/RES-paddle2-Deeplabv3-ResNet50/ --model_filename=model --params_filename=params --save_dir=models/Deeplabv3_ResNet50_fp32 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/Deeplabv3_ResNet50_fp32 --model_filename=model --params_filename=params --save_file=models/Deeplabv3_ResNet50_fp32/Deeplabv3_ResNet50_fp32.onnx
# models/ResNet50_vd_infer/
cd models/ResNet50_vd_infer/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx
cd -
# models/MobileNetV3_large_x1_0_infer/
cd models/MobileNetV3_large_x1_0_infer/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx
cd -
# models/PPHGNet_tiny_infer/
cd models/PPHGNet_tiny_infer/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx
cd -
# models/PPLCNetV2_base_infer/
cd models/PPLCNetV2_base_infer/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx
cd -
# models/EfficientNetB0_infer/
cd models/EfficientNetB0_infer/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx
cd -
## nlp
# models/AFQMC
cd models/AFQMC
paddle2onnx --model_dir ./  --model_filename infer.pdmodel --params_filename infer.pdiparams --save_file model.onnx
cd -
# models/afqmc
cd models/afqmc
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx
cd -
# models/x2paddle_cola
cd models/x2paddle_cola
paddle2onnx --model_dir ./  --model_filename model.pdmodel --params_filename model.pdiparams --save_file model.onnx
cd -



# ================================ INT8 ======================================
# PPYOLOE+ no nms
paddle2onnx --model_dir=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant/ppyoloe_plus_crn_s_80e_coco_no_nms_quant.onnx --deploy_backend='tensorrt' --save_calibration_file=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant/calibration.cache
# PicoDet no nms
paddle2onnx --model_dir=models/picodet_s_416_coco_npu_no_postprocess_quant/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/picodet_s_416_coco_npu_no_postprocess_quant/picodet_s_416_coco_npu_no_postprocess_quant.onnx --deploy_backend='tensorrt' --save_calibration_file=models/picodet_s_416_coco_npu_no_postprocess_quant/calibration.cache
# YOLOv5s
paddle2onnx --model_dir=models/yolov5s_quant/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/yolov5s_quant/yolov5s_quant.onnx --deploy_backend='tensorrt' --save_calibration_file=models/yolov5s_quant/calibration.cache
# YOLOv6s
paddle2onnx --model_dir=models/yolov6s_quant/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/yolov6s_quant/yolov6s_quant.onnx --deploy_backend='tensorrt' --save_calibration_file=models/yolov6s_quant/calibration.cache
# YOLOv7
paddle2onnx --model_dir=models/yolov7_quant/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/yolov7_quant/yolov7_quant.onnx --deploy_backend='tensorrt' --save_calibration_file=models/yolov7_quant/calibration.cache
# PP-HumanSeg-Lite
python utils/paddle_infer_shape.py --model_dir=models/pp_humanseg_qat/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_dir=models/pp_humanseg_int8 --input_shape_dict="{'x':[1, 3, 398, 224]}"
paddle2onnx --model_dir=models/pp_humanseg_int8 --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/pp_humanseg_int8/pp_humanseg_int8.onnx --deploy_backend='tensorrt' --save_calibration_file=models/pp_humanseg_int8/calibration.cache
# PP-Liteseg
python utils/paddle_infer_shape.py --model_dir=models/pp_liteseg_qat/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_dir=models/pp_liteseg_int8 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/pp_liteseg_int8/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/pp_liteseg_int8/pp_liteseg_int8.onnx --deploy_backend='tensorrt' --save_calibration_file=models/pp_liteseg_int8/calibration.cache
# HRNet
python utils/paddle_infer_shape.py --model_dir=models/hrnet_qat/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_dir=models/hrnet_int8 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/hrnet_int8 --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/hrnet_int8/hrnet_int8.onnx --deploy_backend='tensorrt' --save_calibration_file=models/hrnet_int8/calibration.cache
# UNet
python utils/paddle_infer_shape.py --model_dir=models/unet_qat/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_dir=models/unet_int8 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/unet_int8 --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/unet_int8/unet_int8.onnx --deploy_backend='tensorrt' --save_calibration_file=models/unet_int8/calibration.cache
# Deeplabv3-ResNet50
python utils/paddle_infer_shape.py --model_dir=models/deeplabv3_qat/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_dir=models/deeplabv3_int8 --input_shape_dict="{'x':[1, 3, 1024, 2048]}"
paddle2onnx --model_dir=models/deeplabv3_int8 --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=models/deeplabv3_int8/deeplabv3_int8.onnx --deploy_backend='tensorrt' --save_calibration_file=models/deeplabv3_int8/calibration.cache

## classification
# models/ResNet50_vd_QAT
cd models/ResNet50_vd_QAT/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
# models/MobileNetV3_large_x1_0_QAT/
cd models/MobileNetV3_large_x1_0_QAT/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
# models/PPLCNetV2_base_QAT/
cd models/PPLCNetV2_base_QAT/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
# models/PPHGNet_tiny_QAT/
cd models/PPHGNet_tiny_QAT/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
# models/EfficientNetB0_QAT/
cd models/EfficientNetB0_QAT/
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
## nlp
# models/save_ernie3_afqmc_new_cablib
cd models/save_ernie3_afqmc_new_cablib
paddle2onnx --model_dir ./  --model_filename infer.pdmodel --params_filename infer.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
# models/save_ppminilm_afqmc_new_calib
cd models/save_ppminilm_afqmc_new_calib
paddle2onnx --model_dir ./  --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
# models/x2paddle_cola_new_calib
cd models/x2paddle_cola_new_calib
paddle2onnx --model_dir ./  --model_filename model.pdmodel --params_filename model.pdiparams --save_file model.onnx  --deploy_backend tensorrt
cd -
