# encoding: utf-8
"""
case_start.py:
"""
import os
import logging
import yaml

logger = logging.getLogger("paddlerec-log")


class PaddleRecCaseStart(object):
    """
    PaddleRecCaseStart:
    """
    def __init__(self):
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        self.reponame = os.environ["reponame"]
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.step = os.environ["step"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.set_cuda = os.environ["set_cuda"]
        self.system = os.environ["system"]
    
    def update_yaml_config(self, file_path, old_str, new_str):
        """
        update config yaml:
        """
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
    """
    case_start.py run:
    """
    paddlerec_case_start = PaddleRecCaseStart()
    currnet_step = paddlerec_case_start.case_step
    current_name = paddlerec_case_start.case_name
    system = paddlerec_case_start.system
    rd_yaml = os.path.join(paddlerec_case_start.REPO_PATH, paddlerec_case_start.rd_yaml_path)
    qa_yaml = paddlerec_case_start.qa_yaml_name
    if current_name == "dy2st":
        paddlerec_case_start.update_yaml_config(rd_yaml, "#model_init_path:", "model_init_path:")
        paddlerec_case_start.update_yaml_config(rd_yaml, "# model_init_path:", "model_init_path:")
        

if __name__ == "__main__":
    run()
