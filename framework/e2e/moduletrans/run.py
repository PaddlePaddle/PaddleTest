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
import controller


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
                "wget -q --no-proxy "
                "https://paddle-qa.bj.bcebos.com/luozeyu01/framework_e2e_LayerTest/{}/ground_truth.tar".format(self.env)
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

    def upload_resource(self):
        """upload resource"""
        self.prepare()
        os.system("wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz")
        os.system("tar -xzf bos_new.tar.gz")
        final_dict = self.case_set()
        fail_upload_dict = {}
        fail_upload_list = []
        for k, v in final_dict.items():
            for yaml, case_list in v.items():
                fail_upload_dict[yaml] = []
                for case_name in case_list:
                    yml = YamlLoader(yaml)
                    case = yml.get_case_info(case_name)
                    test = controller.ControlModuleTrans(case=case)
                    try:
                        test.mk_dygraph_train_ground_truth()
                        test.mk_dygraph_predict_ground_truth()
                        test.mk_static_train_ground_truth()
                        test.mk_static_predict_ground_truth()
                    except Exception:
                        fail_upload_dict[yaml].append(case_name)
                        fail_upload_list.append(case_name)
        os.system("tar -czf ground_truth.tar ground_truth")
        os.system(
            "python BosClient.py ground_truth.tar paddle-qa/luozeyu01/framework_e2e_LayerTest/{} "
            "https://paddle-qa.bj.bcebos.com/luozeyu01/framework_e2e_LayerTest/{}".format(self.env, self.env)
        )
        print("fail upload dict is: ", fail_upload_dict)
        print("fail upload list is: ", fail_upload_list)

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
    execute = ModuleSystemTest(env="Mac_0", repo_list=["Det"])
    execute.upload_resource()  # baseline上传
    # execute.prepare()
    # execute.run()
