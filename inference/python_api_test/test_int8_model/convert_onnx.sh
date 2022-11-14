python -m pip install paddle2onnx==1.0.2
# ================================ FP32 ======================================
# PPYOLOE-l
paddle2onnx --model_dir=models/ppyoloe_crn_l_300e_coco/ --save_file=models/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --model_filename=model.pdmodel --params_filename=model.pdiparams


# ================================ INT8 ======================================
# PPYOLOE-l
paddle2onnx --model_dir=models/ppyoloe_crn_l_300e_coco_quant/ --model_filename=model.pdmodel --params_filename=model.pdiparams --save_file=ppyoloe_s_quant_416 --deploy_backend='tensorrt'
