import os
import yaml
import wget
import tarfile
import zipfile
import logging

logger = logging.getLogger("paddleslim-log")

class PaddleSlim_Case_Start(object):
    def __init__(self):
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        self.reponame = os.environ["reponame"]
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.step = os.environ["step"]
        # PaddleSlim: step = 'train:single,multi+eval:single,multi'
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]


def run():
    paddleslim_case_start = PaddleSlim_Case_Start()
    current_path = os.getcwd()
    currnet_step = paddleslim_case_start.case_step
    current_name = paddleslim_case_start.case_name

    rd_yaml = os.path.join(paddleslim_case_start.REPO_PATH, paddleslim_case_start.rd_yaml_path)
    qa_yaml = paddleslim_case_start.qa_yaml_name
    with open(rd_yaml, "r") as f:
        content = yaml.load(f, Loader=yaml.FullLoader)

    if currnet_step == "eval" and qa_yaml == "example^auto_compression^nlp^configs^ernie3.0^afqmc":
        content["Global"]["model_dir"] = "./save_afqmc_ERNIE_pruned"
    elif currnet_step == "eval" and qa_yaml == "example^auto_compression^nlp^configs^pp-minilm^auto^afqmc":
        content["Global"]["model_dir"] = "./save_afqmc_pp_minilm_pruned"
    elif currnet_step == "eval" and qa_yaml == "example^post_training_quantization^pytorch_yolo_series^configs^yolov6s_fine_tune":
        if current_name == "single":
            content["model_dir"] = "region_ptq_out"
        else:
            content["model_dir"] = "layer_ptq_out"

    else:
        logger.info("******* {} no update required".format(rd_yaml))

    with open(rd_yaml, "w") as f:
        yaml.dump(content, f)

if __name__ == "__main__":
    run()