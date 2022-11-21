export CUDA_VISIBLE_DEVICES=0
export FLAGS_call_stack_level=2
PYTHON="python"



# ResNet_vd trt int8
echo "[Benchmark] Run ResNet_vd trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT/model.onnx --deploy_backend=tensorrt --input_name=inputs --precision=int8 --calibration_file=models/ResNet50_vd_QAT/calibration.cache --model_name=ResNet_vd
rm -rf model_int8_model.trt 
# MobileNetV3_large trt int8
echo "[Benchmark] Run MobileNetV3_large trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT/model.onnx --deploy_backend=tensorrt --input_name=inputs --precision=int8 --calibration_file=models/MobileNetV3_large_x1_0_QAT/calibration.cache --model_name=MobileNetV3_large
rm -rf model_int8_model.trt
# PPLCNetV2 trt int8
echo "[Benchmark] Run PPLCNetV2 trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT/model.onnx --deploy_backend=tensorrt --precision=int8 --calibration_file=models/PPLCNetV2_base_QAT/calibration.cache --model_name=PPLCNetV2
rm -rf model_int8_model.trt
# PPHGNet_tiny trt int8
echo "[Benchmark] Run PPHGNet_tiny trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT/model.onnx --deploy_backend=tensorrt --precision=int8 --calibration_file=models/PPHGNet_tiny_QAT/calibration.cache  --model_name=PPHGNet_tiny
rm -rf model_int8_model.trt
# EfficientNetB0 trt int8
echo "[Benchmark] Run EfficientNetB0 trt int8"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT/model.onnx --deploy_backend=tensorrt --precision=int8 --calibration_file=models/EfficientNetB0_QAT/calibration.cache --model_name=EfficientNetB0
rm -rf model_int8_model.trt




# ERNIE 3.0-Medium nv-trt int8
echo "[Benchmark] Run NV-TRT ERNIE 3.0-Medium trt int8"
$PYTHON  test_nlp_infer.py --model_path=models/save_ernie3_afqmc_new_cablib/model.onnx --deploy_backend=tensorrt  --task_name='afqmc' --precision=int8 --calibration_file=models/save_ernie3_afqmc_new_cablib/calibration.cache --model_name=ERNIE_3.0-Medium
rm -rf model_int8_model.trt 
# PP-MiniLM nv-trt int8
echo "[Benchmark] Run NV-TRT PP-MiniLM trt int8"
$PYTHON test_nlp_infer.py --model_path=models/save_ppminilm_afqmc_new_calib/model.onnx --deploy_backend=tensorrt  --task_name='afqmc' --precision=int8 --calibration_file=models/save_ppminilm_afqmc_new_calib/calibration.cache --model_name=PP-MiniLM
rm -rf model_int8_model.trt 
# BERT Base nv-trt int8
echo "[Benchmark] Run NV-TRT BERT Base trt int8"
$PYTHON  test_bert_infer.py --model_path=models/x2paddle_cola_new_calib/model.onnx --precision=int8 --batch_size=1 --deploy_backend=tensorrt  --calibration_file=./models/x2paddle_cola_new_calib/calibration.cache --model_name=BERT_Base
rm -rf model_int8_model.trt 