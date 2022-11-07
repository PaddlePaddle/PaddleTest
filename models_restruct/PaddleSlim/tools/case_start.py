import os
import yaml
import wget
import tarfile

class PaddleSlim_Start(object):
    def __init__(self):
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        self.reponame = os.environ["reponame"]
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)

    def change_yaml(self, yaml_path):
        with open(os.path.join(self.REPO_PATH, yaml_path), "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        content["TrainConfig"]["train_iter"] = 50
        content["TrainConfig"]["eval_iter"] = 10
        with open(os.path.join(self.REPO_PATH, yaml_path), "w") as f:
            yaml.dump(content, f)
        return 0

    def wget_and_tar(self, wget_url):
        print("*******")
        print(os.getcwd())

        tar_name = wget_url.split("/")[-1]
        wget.download(wget_url)
        tf = tarfile.open(tar_name)
        tf.extractall(os.getcwd())


def run(function_name):
    model_paddleslim = PaddleSlim_Start()

    print("*******")
    print(function_name)
    print(model_paddleslim.qa_yaml_name)
    print(model_paddleslim.rd_yaml_path)

    model_paddleslim.wget_and_tar('https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar')

    model_paddleslim.change_yaml(model_paddleslim.rd_yaml_path)


if __name__ == "__main__":
    run()