export FLAGS_call_stack_level=2
PYTHON="python"

# PPYOLOE MKLDNN int8
echo "[Benchmark] Run PPYOLOE MKLDNN int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --device=CPU --use_mkldnn=True --cpu_threads=10 --precision=int8
# PicoDet MKLDNN int8
echo "[Benchmark] Run PicoDet MKLDNN int8"
$PYTHON test_ppyoloe_infer.py --model_path=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --precision=int8
# YOLOv5s MKLDNN int8
echo "[Benchmark] Run YOLOv5s MKLDNN int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov5s_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv5 --precision=int8
# YOLOv6s MKLDNN int8
echo "[Benchmark] Run YOLOv6s MKLDNN int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov6s_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv6 --precision=int8
# YOLOv7 MKLDNN int8
echo "[Benchmark] Run YOLOv7 MKLDNN int8"
$PYTHON test_yolo_series_infer.py --model_path=models/yolov7_quant --device=CPU --use_mkldnn=True --cpu_threads=10 --arch=YOLOv7 --precision=int8

# ResNet_vd MKLDNN int8
echo "[Benchmark] Run ResNet_vd MKLDNN int8"
$PYTHON test_image_classification_infer.py --model_path=models/ResNet50_vd_QAT --cpu_num_threads=10 --use_mkldnn=True --use_int8=True
# MobileNetV3_large MKLDNN int8
echo "[Benchmark] Run MobileNetV3_large MKLDNN int8"
$PYTHON test_image_classification_infer.py --model_path=models/MobileNetV3_large_x1_0_QAT --cpu_num_threads=10 --use_mkldnn=True --use_int8=True
# PPLCNetV2 MKLDNN int8
echo "[Benchmark] Run PPLCNetV2 MKLDNN int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPLCNetV2_base_QAT --cpu_num_threads=10 --use_mkldnn=True --use_int8=True
# PPHGNet_tiny MKLDNN int8
echo "[Benchmark] Run PPHGNet_tiny MKLDNN int8"
$PYTHON test_image_classification_infer.py --model_path=models/PPHGNet_tiny_QAT --cpu_num_threads=10 --use_mkldnn=True --use_int8=True
# EfficientNetB0 MKLDNN int8
echo "[Benchmark] Run EfficientNetB0 MKLDNN int8"
$PYTHON test_image_classification_infer.py --model_path=models/EfficientNetB0_QAT --cpu_num_threads=10 --use_mkldnn=True --use_int8=True

# PP-HumanSeg-Lite MKLDNN int8
echo "[Benchmark] Run PP-HumanSeg-Lite MKLDNN int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_humanseg_qat --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --device=CPU --use_mkldnn=True --precision=int8 --cpu_threads=10
# PP-Liteseg MKLDNN int8
echo "[Benchmark] Run PP-Liteseg MKLDNN int8"
$PYTHON test_segmentation_infer.py --model_path=models/pp_liteseg_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --precision=int8 --cpu_threads=10
# HRNet MKLDNN int8
echo "[Benchmark] Run HRNet MKLDNN int8"
$PYTHON test_segmentation_infer.py --model_path=models/hrnet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --precision=int8 --cpu_threads=10
# UNet MKLDNN int8
echo "[Benchmark] Run UNet MKLDNN int8"
$PYTHON test_segmentation_infer.py --model_path=models/unet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --precision=int8 --cpu_threads=10
# Deeplabv3-ResNet50 MKLDNN int8
echo "[Benchmark] Run Deeplabv3-ResNet50 MKLDNN int8"
$PYTHON test_segmentation_infer.py --model_path=models/deeplabv3_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --device=CPU --use_mkldnn=True --precision=int8 --cpu_threads=10

# ERNIE 3.0-Medium MKLDNN int8
echo "[Benchmark] Run ERNIE 3.0-Medium MKLDNN int8"
$PYTHON test_nlp_infer.py --model_path=models/save_ernie3_afqmc_new_cablib --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --device=cpu --use_mkldnn=True --cpu_threads=10 --precision=int8
# PP-MiniLM MKLDNN int8
echo "[Benchmark] Run PP-MiniLM MKLDNN int8"
$PYTHON test_nlp_infer.py --model_path=models/save_ppminilm_afqmc_new_calib --task_name='afqmc' --device=cpu --use_mkldnn=True --cpu_threads=10 --precision=int8
# BERT Base MKLDNN int8
echo "[Benchmark] Run BERT Base MKLDNN int8"
$PYTHON test_bert_infer.py --model_path=models/x2paddle_cola_new_calib --device=cpu --use_mkldnn=True --cpu_threads=10 --batch_size=1 --precision=int8
