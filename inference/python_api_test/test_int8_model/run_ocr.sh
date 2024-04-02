export PYTHONPATH=$PWD/PaddleOCR:$PYTHONPATH

# 测速
# paddle2onnx --model_dir=models/PPOCRV3_det_QAT --model_filename=inference.pdmodel --params_filename=inference.pdiparams --save_file=models/PPOCRV3_det_QAT/model.onnx --save_calibration_file=models/PPOCRV3_det_QAT/calibration.cache --deploy_backend='tensorrt'
python test_ocr_infer.py --model_type='det' --model_path="./models/PPOCRV3_det_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --image_file="test.jpg" --device='GPU' --use_trt=True --precision='int8' --benchmark=True --deploy_backend='paddle_inference'

# 测精度
# python test_ocr_infer.py --model_path="./PPOCRV3_det_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --dataset_config="./configs/ppocrv3_det.yaml" --device='GPU' --use_trt=True --precision='int8' --deploy_backend='paddle_inference'


# 测速 ocr_rec
python test_ocr_infer.py --model_type='rec' --model_path="./PPOCRV3_rec_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --image_file="test.jpg" --device='GPU' --use_trt=True --precision='int8' --benchmark=True --deploy_backend='paddle_inference'

# 测精度 ocr_rec
# python test_ocr_infer.py --model_path="./PPOCRV3_rec_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --dataset_config="./configs/ppocrv3_rec.yaml" --device='GPU' --use_trt=True --precision='int8' --deploy_backend='paddle_inference'
