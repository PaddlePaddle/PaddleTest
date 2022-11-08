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
        return 0
    
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
        return 0
    
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
    print("*******")
    print(paddleslim_start.qa_yaml_name)
    print(paddleslim_start.rd_yaml_path)

    current_path = os.getcwd()
    yaml_path = paddleslim_start.rd_yaml_path
    with open(os.path.join(paddleslim_start.REPO_PATH, yaml_path), "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)

    if paddleslim_start.qa_yaml_name == "example-auto_compression-detection-configs-ppyoloe_l_qat_dis":
        paddleslim_start.wget_and_tar("https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar")
        paddleslim_start.wget_and_zip("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip") 
        content["TrainConfig"]["train_iter"] = 20
        content["TrainConfig"]["eval_iter"] = 10
        content["Global"]["model_dir"] = current_path + "/ppyoloe_crn_l_300e_coco"
    else:
        logger.info("### no exists {} ".format(paddleslim_start.qa_yaml_name))

    with open(os.path.join(paddleslim_start.REPO_PATH, yaml_path), "w") as f:
            yaml.dump(content, f)
    
    reader_yaml_path = paddleslim_start.REPO_PATH + "/example/auto_compression/detection/configs/yolo_reader.yml"
    paddleslim_start.update_yaml_config(reader_yaml_path, "dataset_dir: dataset/coco/", "dataset_dir: " + os.getcwd() + '/coco')
    
    return 0

if __name__ == "__main__":
    run()