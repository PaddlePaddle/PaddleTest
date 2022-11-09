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
    
    def wget_and_zip(self, wget_url):
        zip_name = wget_url.split("/")[-1]
        if os.path.exists(zip_name):
            logger.info("*** {} already download ".format(zip_name))
        else:
            try:
                logger.info("*** {} start download and unzip ".format(zip_name))
                wget.download(wget_url)
                fz = zipfile.ZipFile(zip_name, 'r')
                for file in fz.namelist():
                    fz.extract(file, os.getcwd())
                logger.info("*** {}  end download and unzip ".format(zip_name))
            except:
                logger.info("*** {} start download or unzip failed".format(zip_name))
    
    def wget_and_tar(self, wget_url):
        tar_name = wget_url.split("/")[-1]
        if os.path.exists(tar_name):
            logger.info("*** {} already download ".format(tar_name))
        else:
            try:
                logger.info("*** {} start download and tar -x ".format(tar_name))
                wget.download(wget_url)
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
                logger.info("*** {} end download and tar -x ".format(tar_name))
            except:
                logger.info("*** {} start download or  tar -x failed".format(tar_name))
    
    def update_yaml_config(self, file_path, old_str, new_str):
        file_data = ""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if old_str in line:
                    line = line.replace(old_str, new_str)
                file_data += line
        with open(file_path,"w",encoding="utf-8") as f:
            f.write(file_data)

def run():
    paddleslim_start = PaddleSlim_Start()
    current_path = os.getcwd()
    with open(os.path.join(paddleslim_start.REPO_PATH, paddleslim_start.rd_yaml_path), "r") as f:
        content = yaml.load(f, Loader=yaml.FullLoader)

    if paddleslim_start.qa_yaml_name == "example:auto_compression:detection:configs:ppyoloe_l_qat_dis":
        paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar")
        paddleslim_start.wget_and_zip("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip")
        content["TrainConfig"]["train_iter"] = 20
        content["TrainConfig"]["eval_iter"] = 10
        content["Global"]["model_dir"] = current_path + "/ppyoloe_crn_l_300e_coco"
    elif paddleslim_start.qa_yaml_name == "example:auto_compression:pytorch_yolo_series:configs:yolov5s_qat_dis":
        paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/detection/yolov5s_infer.tar")
        paddleslim_start.wget_and_zip("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip")
        content["TrainConfig"]["train_iter"] = 20
        content["TrainConfig"]["eval_iter"] = 10
        content["Global"]["coco_dataset_dir"] = current_path + "/coco"
    elif paddleslim_start.qa_yaml_name == "example:auto_compression:image_classification:configs:MobileNetV1:qat_dis":
        paddleslim_start.wget_and_tar("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar")
        paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar")
        content["Global"]["data_dir"] = current_path + "/ILSVRC2012"
        content["Global"]["model_dir"] = current_path + "/MobileNetV1_infer"
        content["TrainConfig"]["epochs"] = 1
        content["TrainConfig"]["eval_iter"] = 50
    elif paddleslim_start.qa_yaml_name == "example:auto_compression:image_classification:configs:ResNet50_vd:qat_dis":
        paddleslim_start.wget_and_tar("https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar")
        paddleslim_start.wget_and_tar("https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar")
        content["Global"]["data_dir"] = current_path + "/ILSVRC2012"
        content["Global"]["model_dir"] = current_path + "/ResNet50_vd_infer"
        content["TrainConfig"]["epochs"] = 1
        content["TrainConfig"]["eval_iter"] = 50
    elif paddleslim_start.qa_yaml_name == "example:auto_compression:nlp:configs:ernie3.0:afqmc":
        paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/AFQMC.tar")
        content["Global"]["model_dir"] = current_path + "/AFQMC"
        content["TrainConfig"]["epochs"] = 1
        content["TrainConfig"]["eval_iter"] = 50
    elif paddleslim_start.qa_yaml_name == "example:auto_compression:nlp:configs:pp-minilm:auto:afqmc.yaml":
        paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar")
        content["Global"]["model_dir"] = current_path + "/afqmc"
        content["TrainConfig"]["epochs"] = 1
        content["TrainConfig"]["eval_iter"] = 50
        paddleslim_start.update_yaml_config(paddleslim_start.rd_yaml_path, "HyperParameterOptimization:", "#HyperParameterOptimization:")
    elif paddleslim_start.qa_yaml_name == "example:auto_compression:semantic_segmentation:configs:pp_liteseg:pp_liteseg_qat.yaml" or \
            paddleslim_start.qa_yaml_name == "example:auto_compression:semantic_segmentation:configs:pp_liteseg:pp_liteseg_sparse":
        paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar")
        paddleslim_start.wget_and_zip("https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-PPLIteSegSTDC1.zip")
        content["Global"]["model_dir"] = current_path + "/RES-paddle2-PPLIteSegSTDC1"
        content["TrainConfig"]["epochs"] = 1
        content["TrainConfig"]["eval_iter"] = 50

    else:
        logger.info("### no exists {} ".format(paddleslim_start.qa_yaml_name))

    with open(os.path.join(paddleslim_start.REPO_PATH, paddleslim_start.rd_yaml_path), "w") as f:
        yaml.dump(content, f)
    
    # example:auto_compression:detection:configs:ppyoloe_l_qat_dis 修改数据路径
    ppyoloe_l_qat_dis_reader = paddleslim_start.REPO_PATH + "/example/auto_compression/detection/configs/yolo_reader.yml"
    paddleslim_start.update_yaml_config(ppyoloe_l_qat_dis_reader, "dataset_dir: dataset/coco/", "dataset_dir: " + current_path + '/coco')

    # auto_compression：semantic_segmentation demo 修改数据路径
    semantic_segmentation_reader = "example/auto_compression/semantic_segmentation/configs/dataset/cityscapes_1024x512_scale1.0.yml"
    with open(os.path.join(paddleslim_start.REPO_PATH, semantic_segmentation_reader), "r") as f_reader:
        content_reader = yaml.load(f_reader, Loader=yaml.FullLoader)
    content_reader["train_dataset"]["dataset_root"] = current_path + "/mini_cityscapes"
    with open(os.path.join(paddleslim_start.REPO_PATH, semantic_segmentation_reader), "w") as f_reader:
        yaml.dump(content_reader, f_reader)


if __name__ == "__main__":
    run()