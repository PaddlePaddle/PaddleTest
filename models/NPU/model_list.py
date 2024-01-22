#!/usr/env/bin python

import sys
import yaml
from pathlib import Path

def run(repo, configs_path, chain_txt, dst_file):
    """
    """
    with open("configs/model.yaml", 'r') as f:
        model_list = yaml.load(f, Loader=yaml.FullLoader)
    if repo not in model_list.keys():
        white = ["all"]
    else:
        white = model_list[repo]["white"].split("|")
        black = model_list[repo]["black"].split("|")

    models = []
    for item in Path(configs_path).rglob(chain_txt):
        file_path = 'test_tipc/' + str(item).split('/test_tipc/')[-1]
        if "all" in white:
            models.append(file_path)
            continue
        flag = False
        for i in white:
            flag = False
            if i in file_path:
                flag = True
                for j in black:
                    if j in file_path:
                        flag = False
            if flag == True:
                models.append(file_path)
                break

    with open(dst_file, 'w') as f:
        for model in models:
            f.write(model + "\n")


if __name__ == "__main__":
    repo = sys.argv[1]
    configs_path = sys.argv[2]
    chain_txt = sys.argv[3]
    dst_file = sys.argv[4]
    run(repo, configs_path, chain_txt, dst_file)
