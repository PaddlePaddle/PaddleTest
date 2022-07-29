#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
run module test
"""

import os
import platform
import pytest
import allure
from yaml_loader import YamlLoader


class ModuleSystemTest(object):
    """test"""

    def __init__(self, env, repo_list):
        """init"""
        self.cur_path = os.getcwd()
        self.report_dir = os.path.join(self.cur_path, "report")
        self.env = env
        self.repo_list = repo_list

    def prepare(self):
        """prepare source"""

        if not os.path.exists(os.path.join(self.cur_path, "ground_truth.tar")):
            os.system(
                "wget https://paddle-qa.bj.bcebos.com/luozeyu01/framework_e2e_LayerTest/{}/ground_truth.tar".format(
                    self.env
                )
            )
            os.system("tar -xzf ground_truth.tar")

        if "Det" in self.repo_list and not os.path.exists(os.path.join(self.cur_path, "ppdet")):
            if not os.path.exists(os.path.join(self.cur_path, "ppdet")):
                os.system("git clone -b develop https://github.com/PaddlePaddle/PaddleDetection.git")
                os.system("cp -r PaddleDetection/ppdet .")
                os.system("cd PaddleDetection && python -m pip install -r requirements.txt")
                os.system("cd PaddleDetection && python setup.py install")

        if "Clas" in self.repo_list and not os.path.exists(os.path.join(self.cur_path, "ppcls")):
            if not os.path.exists(os.path.join(self.cur_path, "ppcls")):
                os.system("git clone -b develop https://github.com/PaddlePaddle/PaddleClas.git")
                os.system("cp -r PaddleClas/ppcls .")
                os.system("cd PaddleClas && python -m pip install -r requirements.txt")
                os.system("cd PaddleClas && python setup.py install")

    def case_set(self):
        """all cases set"""
        yaml_dict = {}
        for repo in self.repo_list:
            yaml_path = os.path.join(self.cur_path, "yaml", repo)
            repo_yaml_list = self.get_all_yaml(base_path=yaml_path)
            yaml_dict[repo] = repo_yaml_list

        final_dict = {}
        for k, v in yaml_dict.items():
            final_dict[k] = {}
            for y in v:
                yml = YamlLoader(y)
                all_cases_list = []
                all_cases_dict = yml.get_all_case_name()
                for key in all_cases_dict:
                    all_cases_list.append(key)
                final_dict[k][y] = all_cases_list

        return final_dict

    def get_all_yaml(self, base_path, all_yaml=[]):
        """show all file.yml"""
        file_list = os.listdir(base_path)

        for file in file_list:
            yaml_path = os.path.join(base_path, file)

            if os.path.isdir(yaml_path):
                self.get_all_yaml(yaml_path, all_yaml)
            else:
                if not file.endswith(".yml"):
                    continue
                else:
                    all_yaml.append(yaml_path)
        return all_yaml

    def run(self):
        """run test"""
        final_dict = self.case_set()
        for k, v in final_dict.items():
            for yaml, case_list in v.items():
                for case in case_list:
                    if platform.system() == "Windows":
                        os.system(
                            "python.exe -m pytest -sv test_run.py --yaml={} --case={} --alluredir={}".format(
                                yaml, case, self.report_dir
                            )
                        )
                    else:
                        os.system(
                            "python -m pytest -sv test_run.py --yaml={} --case={} --alluredir={}".format(
                                yaml, case, self.report_dir
                            )
                        )


if __name__ == "__main__":
    execute = ModuleSystemTest(env="cuda102", repo_list=["Det"])
    execute.prepare()
    execute.run()
