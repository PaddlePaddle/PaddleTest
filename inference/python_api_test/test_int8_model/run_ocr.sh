export PYTHONPATH=$PWD/PaddleOCR:$PYTHONPATH

# 测速 ocr_det
python3.7 infer.py --model_type='det' --model_path="./PPOCRV3_det_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --image_file="test.jpg" --device='GPU' --use_trt=True --precision='int8' --benchmark=True

# 测精度 ocr_det
# python3.7 infer.py --model_path="./PPOCRV3_det_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --dataset_config="./configs/ppocrv3_det.yaml" --device='GPU' --use_trt=True --precision='int8'

# 测速 ocr_rec
python3.7 infer.py --model_type='rec' --model_path="./PPOCRV3_rec_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --image_file="test.jpg" --device='GPU' --use_trt=True --precision='int8' --benchmark=True

# 测精度 ocr_rec
# python3.7 infer.py --model_path="./PPOCRV3_rec_QAT" --model_filename="inference.pdmodel" --params_filename="inference.pdiparams" --dataset_config="./configs/ppocrv3_rec.yaml" --device='GPU' --use_trt=True --precision='int8'
