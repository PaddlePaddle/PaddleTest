# encoding: utf-8
"""
start.py:
"""
import os
import logging

logger = logging.getLogger("paddlerec-log")


class PaddleRecStart(object):
    """
    PaddleRecStart:
    """

    def __init__(self):
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        self.reponame = os.environ["reponame"]
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)


def run():
    """
    start.py run:
    """
    paddlerec_start = PaddleRecStart()
    current_path = os.getcwd()
    qa_yaml = paddlerec_start.qa_yaml_name

    datasets_path = paddlerec_start.REPO_PATH + "/datasets"
    os.chdir(datasets_path)
    if qa_yaml == "models^contentunderstanding^textcnn^config_bigdata":
        os.chdir("senti_clas")
    elif qa_yaml == "models^match^dssm^config_bigdata":
        os.chdir("BQ_dssm")
    elif qa_yaml == "models^rank^dnn^config_bigdata" or qa_yaml == "models^rank^wide_deep^config_bigdata":
        os.chdir("criteo")
    elif qa_yaml == "models^rank^dnn^config_bigdata":
        os.chdir("criteo")
    elif qa_yaml == "models^multitask^mmoe^config_bigdata" or qa_yaml == "models^rank^wide_deep^config_bigdata":
        os.chdir("census")
    elif qa_yaml == "models^recall^ncf^config_bigdata":
        os.chdir("movielens_pinterest_NCF")
    else:
        logger.info("******* {} not exists".format(qa_yaml))

    os.system("bash run.sh")
    os.chdir(current_path)


if __name__ == "__main__":
    run()
