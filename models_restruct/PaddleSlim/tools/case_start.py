# encoding: utf-8
"""
case_start.py:
"""
import os
import logging
import yaml

logger = logging.getLogger("paddleslim-log")


class PaddleSlimCaseStart(object):
    """
    PaddleSlimCaseStart:
    """

    def __init__(self):
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        self.reponame = os.environ["reponame"]
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.step = os.environ["step"]
        # PaddleSlim: step = 'train:single,multi+eval:single,multi'
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.set_cuda = os.environ["set_cuda"]
        self.system = os.environ["system"]


def run():
    """
    case_start.py run:
    """
    paddleslim_case_start = PaddleSlimCaseStart()
    currnet_step = paddleslim_case_start.case_step
    current_name = paddleslim_case_start.case_name
    system = paddleslim_case_start.system
    os.environ["CUDA_VISIBLE_DEVICES"] = paddleslim_case_start.set_cuda
    set_cuda_single_card = paddleslim_case_start.set_cuda.split(",")[0]
    if current_name == "single":
        os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_single_card
    rd_yaml = os.path.join(paddleslim_case_start.REPO_PATH, paddleslim_case_start.rd_yaml_path)
    qa_yaml = paddleslim_case_start.qa_yaml_name
    if qa_yaml.split("^")[0] != "case":
        with open(rd_yaml, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        if qa_yaml == "example^auto_compression^nlp^configs^ernie3.0^afqmc":
            os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_single_card
            if currnet_step == "eval":
                content["Global"]["model_dir"] = "./save_afqmc_ERNIE_pruned"
        elif qa_yaml == "example^auto_compression^nlp^configs^pp-minilm^auto^afqmc":
            os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_single_card
            if currnet_step == "eval":
                content["Global"]["model_dir"] = "./save_afqmc_pp_minilm_pruned"
        elif (
            currnet_step == "eval"
            and qa_yaml == "example^post_training_quantization^pytorch_yolo_series^configs^yolov6s_fine_tune"
        ):
            if current_name == "single":
                content["model_dir"] = "region_ptq_out"
            else:
                content["model_dir"] = "layer_ptq_out"
        elif (
            qa_yaml == "example^auto_compression^pytorch_yolo_series^configs^yolov5s_qat_dis"
            and current_name == "single"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_single_card
        elif (
            qa_yaml == "example^auto_compression^semantic_segmentation^configs^pp_liteseg^pp_liteseg_sparse"
            and current_name == "single"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_single_card
        elif (
            qa_yaml == "example^full_quantization^image_classification^configs^mobilenetv3_large_qat_dis"
            and system == "windows"
        ):
            content["Global"]["batch_size"] = 16
        else:
            logger.info("******* {} no update required".format(rd_yaml))

        with open(rd_yaml, "w") as f:
            yaml.dump(content, f)
    else:
        logger.info("******* yaml：{} no exists".format(rd_yaml))


if __name__ == "__main__":
    run()
