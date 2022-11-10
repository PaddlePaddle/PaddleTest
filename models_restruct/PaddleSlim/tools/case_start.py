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


def run(function_name):
    paddleslim_case_start = PaddleSlim_Case_Start()
    current_path = os.getcwd()
    with open(os.path.join(paddleslim_case_start.REPO_PATH, paddleslim_case_start.rd_yaml_path), "r") as f:
        content = yaml.load(f, Loader=yaml.FullLoader)

    if paddleslim_case_start.step == "eval" and paddleslim_case_start.qa_yaml_name == "example:auto_compression:nlp:configs:ernie3.0:afqmc.yaml":
        content["Global"]["model_dir"] = "./save_afqmc_ERNIE_pruned"
    elif paddleslim_case_start.step == "eval" and paddleslim_case_start.qa_yaml_name == "example:auto_compression:nlp:configs:pp-minilm:auto:afqmc.yaml":
        content["Global"]["model_dir"] = "./save_afqmc_pp_minilm_pruned"
    
    else:
        logger.info("### {} no update required".format(paddleslim_case_start.rd_yaml_path))


if __name__ == "__main__":
    run()