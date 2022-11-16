import os
import yaml
import wget
import tarfile
import zipfile
import logging

logger = logging.getLogger("paddleslim-log")

class PaddleSlim_Start(object):
    def __init__(self):
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        self.reponame = os.environ["reponame"]
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.set_cuda = os.environ["set_cuda"]

    
    def wget_and_zip(self, wget_url):
        zip_name = wget_url.split("/")[-1]
        if os.path.exists(zip_name):
            return 
            # logger.info("******* {} already download ".format(zip_name))
        else:
            try:
                logger.info("******* {} start download and unzip ".format(zip_name))
                wget.download(wget_url)
                fz = zipfile.ZipFile(zip_name, 'r')
                for file in fz.namelist():
                    fz.extract(file, os.getcwd())
                logger.info("******* {}  end download and unzip ".format(zip_name))
            except:
                logger.info("******* {} start download or unzip failed".format(zip_name))
    
    def wget_and_tar(self, wget_url):
        tar_name = wget_url.split("/")[-1]
        if os.path.exists(tar_name):
            return
            #logger.info("******* {} already download ".format(tar_name))
        else:
            try:
                logger.info("******* {} start download and tar -x ".format(tar_name))
                wget.download(wget_url)
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
                logger.info("******* {} end download and tar -x ".format(tar_name))
            except:
                logger.info("******* {} start download or  tar -x failed".format(tar_name))

    def wget_and_files(self,wget_url):
        file_name = wget_url.split("/")[-1]
        if os.path.exists(file_name):
            return
            #logger.info("******* {} already download ".format(file_name))
        else:
            try:
                logger.info("******* {} start download ".format(file_name))
                wget.download(wget_url)
                logger.info("******* {} end download  ".format(file_name))
            except:
                logger.info("******* {} start download failed".format(file_name))
    
    def update_yaml_config(self, file_path, old_str, new_str):
        logger.info("******* {} is updating".format(file_path))
        file_data = ""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if old_str in line:
                    line = line.replace(old_str, new_str)
                file_data += line
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_data)

def run():
    paddleslim_start = PaddleSlim_Start()
    current_path = os.getcwd()
    rd_yaml = os.path.join(paddleslim_start.REPO_PATH, paddleslim_start.rd_yaml_path)
    qa_yaml = paddleslim_start.qa_yaml_name
    os.environ["CUDA_VISIBLE_DEVICES"] = paddleslim_start.set_cuda
    set_cuda_single_card = paddleslim_start.set_cuda.split(",")[0]
    
    if qa_yaml.split("^")[0] != "case":
        with open(rd_yaml, "r", encoding="utf-8") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)

        if qa_yaml == "example^auto_compression^detection^configs^ppyoloe_l_qat_dis":
            paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar")
            paddleslim_start.wget_and_zip("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip")
            content["TrainConfig"]["train_iter"] = 50
            content["TrainConfig"]["eval_iter"] = 10
            content["Global"]["model_dir"] = current_path + "/ppyoloe_crn_l_300e_coco"
            # example/auto_compression:detection:configs:ppyoloe_l_qat_dis 修改数据路径
            ppyoloe_l_qat_dis_reader = paddleslim_start.REPO_PATH + "/example/auto_compression/detection/configs/yolo_reader.yml"
            paddleslim_start.update_yaml_config(ppyoloe_l_qat_dis_reader, "dataset/coco/", current_path + '/coco')
        elif qa_yaml == "example^auto_compression^pytorch_yolo_series^configs^yolov5s_qat_dis":
            paddleslim_start.wget_and_files("https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx")
            paddleslim_start.wget_and_zip("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip")
            content["TrainConfig"]["train_iter"] = 50
            content["TrainConfig"]["eval_iter"] = 10
            content["Global"]["model_dir"] = current_path + "/yolov5s.onnx"
            content["Global"]["coco_dataset_dir"] = current_path + "/coco"
        elif qa_yaml == "example^auto_compression^image_classification^configs^MobileNetV1^qat_dis":
            paddleslim_start.wget_and_tar("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar")
            paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar")
            content["Global"]["data_dir"] = current_path + "/ILSVRC2012"
            content["Global"]["model_dir"] = current_path + "/MobileNetV1_infer"
            content["TrainConfig"]["epochs"] = 2
            content["TrainConfig"]["eval_iter"] = 50
        elif qa_yaml == "example^auto_compression^image_classification^configs^ResNet50_vd^qat_dis":
            paddleslim_start.wget_and_tar("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar")
            paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar")
            content["Global"]["data_dir"] = current_path + "/ILSVRC2012"
            content["Global"]["model_dir"] = current_path + "/ResNet50_vd_infer"
            content["TrainConfig"]["epochs"] = 2
            content["TrainConfig"]["eval_iter"] = 50
        elif qa_yaml == "example^auto_compression^nlp^configs^ernie3.0^afqmc":
            os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_single_card
            paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/AFQMC.tar")
            content["Global"]["model_dir"] = current_path + "/AFQMC"
            content["TrainConfig"]["epochs"] = 1
            content["TrainConfig"]["eval_iter"] = 50
        elif qa_yaml == "example^auto_compression^nlp^configs^pp-minilm^auto^afqmc":
            os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_single_card
            paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar")
            content["Global"]["model_dir"] = current_path + "/afqmc"
            content["TrainConfig"]["epochs"] = 1
            content["TrainConfig"]["eval_iter"] = 50
            #HyperParameterOptimization、QuantPost 会导致训练时间过长，先删除该配置；
            del content["HyperParameterOptimization"]
            del content["QuantPost"]
            #paddleslim_start.update_yaml_config(rd_yaml, "HyperParameterOptimization:", "#HyperParameterOptimization:")
        elif qa_yaml == "example^auto_compression^semantic_segmentation^configs^pp_liteseg^pp_liteseg_qat" or \
                qa_yaml == "example^auto_compression^semantic_segmentation^configs^pp_liteseg^pp_liteseg_sparse":
            paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar")
            paddleslim_start.wget_and_zip("https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-PPLIteSegSTDC1.zip")
            content["Global"]["model_dir"] = current_path + "/RES-paddle2-PPLIteSegSTDC1"
            content["TrainConfig"]["epochs"] = 2
            content["TrainConfig"]["eval_iter"] = 50
        elif qa_yaml == "example^full_quantization^image_classification^configs^mobilenetv3_large_qat_dis":
            paddleslim_start.wget_and_tar("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar")
            paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar")
            content["Global"]["model_dir"] = current_path + "/MobileNetV3_large_x1_0_infer"
            content["Global"]["data_dir"] = current_path + "/ILSVRC2012"
            content["TrainConfig"]["epochs"] = 2
            content["TrainConfig"]["eval_iter"] = 50
        elif qa_yaml == "example^full_quantization^picodet^configs^picodet_npu_with_postprocess":
            paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar")
            paddleslim_start.wget_and_zip("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip")
            content["Global"]["model_dir"] = current_path + "/picodet_s_416_coco_npu"
            content["Global"]["data_dir"] = current_path + "/ILSVRC2012"
            content["TrainConfig"]["train_iter"] = 80
            content["TrainConfig"]["eval_iter"] = 10
            # full_quantization/picodet demo 修改数据路径
            full_quant_picodet_reader = paddleslim_start.REPO_PATH + "/example/full_quantization/picodet/configs/picodet_reader.yml"
            paddleslim_start.update_yaml_config(full_quant_picodet_reader, "dataset_dir: dataset/coco/", "dataset_dir: " + current_path + '/coco')
        elif qa_yaml == "example^post_training_quantization^pytorch_yolo_series^configs^yolov6s_fine_tune":
            paddleslim_start.wget_and_files("https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx")
            paddleslim_start.wget_and_zip("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip")
            content["model_dir"] = current_path + "/yolov6s.onnx"
            content["dataset_dir"] = current_path + "/coco"
        else:
            logger.info("### no exists {} ".format(qa_yaml))

        with open(rd_yaml, "w", encoding="utf-8") as f:
            yaml.dump(content, f)

        # auto_compression/semantic_segmentation demo 修改数据路径
        semantic_segmentation_reader = paddleslim_start.REPO_PATH + "/example/auto_compression/semantic_segmentation/configs/dataset/cityscapes_1024x512_scale1.0.yml"
        with open(semantic_segmentation_reader, "r") as f_reader:
            content_reader = yaml.load(f_reader, Loader=yaml.FullLoader)
        content_reader["train_dataset"]["dataset_root"] = current_path + "/mini_cityscapes"
        content_reader["val_dataset"]["dataset_root"] = current_path + "/mini_cityscapes"
        with open(semantic_segmentation_reader, "w") as f_reader:
            yaml.dump(content_reader, f_reader)
    
    else:
        update_count = 0
        if update_count == 0:
            demo_path = paddleslim_start.REPO_PATH + "/demo"
            print("demo_path:" + demo_path)
            os.chdir(demo_path)
            if not os.path.exists("data"):
                os.mkdir("data")
            os.chdir("data")
            print("date_path:" + os.getcwd())
            paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar")
            # demo/quant/quant_embedding 数据集
            paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/convert_text8.tar")
            paddleslim_start.wget_and_files("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/test_build_dict")
            paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/test_mid_dir.tar")
            paddleslim_start.wget_and_files("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/test_build_dict_word_to_id")
            os.chdir(demo_path)
            if not os.path.exists("pretrain"):
                os.mkdir("pretrain")
            os.chdir("pretrain")
            print("pretrain_path:" + os.getcwd())
            for model in ["MobileNetV1","MobileNetV2","MobileNetV3_large_x1_0_ssld","ResNet101_vd","ResNet34","ResNet50","ResNet50_vd"]:
                wget_url = "http://paddle-imagenet-models-name.bj.bcebos.com/" + model + "_pretrained.tar"
                paddleslim_start.wget_and_tar(wget_url)
                
            paddleslim_start.wget_and_tar("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar")
            # 更新imagenet_reader.py 的ILSVRC2012 路径
            os.chdir(demo_path+"/dygraph/pruning")
            if not os.path.exists("data"):
                os.mkdir("data")
            os.chdir("data")
            paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar")
            update_count += 1
            os.chdir(current_path)
        

if __name__ == "__main__":
    run()
