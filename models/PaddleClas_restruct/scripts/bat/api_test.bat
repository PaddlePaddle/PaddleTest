#whl包的测试

# det) #没有训练过程，只有预测过程放到api test中
#     mkdir -p models
#     # 下载通用检测 inference 模型并解压
#     wget -P ./models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
#     tar -xf ./models/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar -C ./models/
#     python python/predict_det.py -c configs/inference_det.yaml
# ;;
