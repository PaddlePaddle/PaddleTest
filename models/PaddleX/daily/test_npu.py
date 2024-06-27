import os
def run_model(model_name, model_cls, data_dir, env, concurrency=1):
    os.system(f"python main.py -c paddlex/configs/{model_cls}/{model_name}.yaml -o Global.mode=train -o Global.dataset_dir=${data_dir} -o Global.device={}:12,13,14,15 >test_${model_name}.log 2>&1")
    return 0



def test_npu():
    with open('./model_npu.json', 'r', encoding='utf-8') as f:
    # 读取JSON数据
        test_json = json.load(f)
        # 遍历 JSON 对象获取所有的键
    for key in test_json:
        model_name = key
        model_cls = test_json[key]
        env = "npu"
        if model_cls == "image_classification":
            if not os.path.exists("cls_flowers_examples.tar"):
                os.system("wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar")
                os.system("tar -xvf cls_flowers_examples.tar")
            data_dir = "./cls_flowers_examples"
        elif model_cls == "object_detection":
            if not os.path.exists("det_coco_examples.tar"):
                os.system("wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar")
                os.system("tar -xvf det_coco_examples.tar")
            data_dir = "./det_coco_examples"
        elif model_cls == "semantic_segmentation":
            if not os.path.exists("seg_optic_examples.tar"):
                os.system("wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar")
                os.system("tar -xvf seg_optic_examples.tar")
            data_dir = "./seg_optic_examples"
        elif model_cls == "ts_forecast":
            if not os.path.exists("ts_anomaly_examples.tar"):
                os.system("wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar")
                os.system("tar -xvf ts_anomaly_examples.tar")
            data_dir = "./ts_anomaly_examples"
        run_model(model_name, model_cls, data_dir, env)

if __name__ == '__main__':
    test_npu()