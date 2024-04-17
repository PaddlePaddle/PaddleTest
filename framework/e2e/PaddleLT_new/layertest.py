#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
测试执行器
"""
import os
import traceback
from engine.engine_map import engine_map
from strategy.compare import base_compare
from tools.yaml_loader import YamlLoader
from tools.logger import Logger


class LayerTest(object):
    """
    单个Layer case 执行
    """

    def __init__(self, title, layerfile, testing):
        """ """
        self.title = title

        self.layerfile = layerfile.replace(".py", "").replace("/", ".").lstrip(".")

        # 解析testing.yml
        self.test_config = YamlLoader(yml=testing)
        self.testings = self.test_config.yml.get("testings")
        self.testings_list = self.test_config.get_junior_name("testings")

        self.compare_list = self.test_config.yml.get("compare")

        self.logger = Logger("PaddleLT")
        self.report_dir = os.path.join(os.getcwd(), "report")

    def _single_run(self, testing, layerfile):
        """
        单次执行器测试
        :param testing: 'dy_train', 'dy_eval'...
        :return:
        """
        layer_test = engine_map[testing](testing=self.testings.get(testing), layerfile=layerfile)

        res = getattr(layer_test, testing)()
        return res

    def _case_run(self):
        """
        用于单个子图精度测试
        """
        exc = 0
        res_dict = {}
        compare_res_list = []
        self.logger.get_log().info("测试case名称: {}".format(self.title))
        for testing in self.testings_list:
            try:
                self.logger.get_log().info("测试执行器: {}".format(testing))
                res = self._single_run(testing=testing, layerfile=self.layerfile)
                res_dict[testing] = res
            except Exception:
                bug_trace = traceback.format_exc()
                exc += 1
                res_dict[testing] = bug_trace
                self.logger.get_log().warn("执行器异常结果: {}".format(bug_trace))

        for comparing in self.compare_list:
            tmp = {}
            latest = comparing.get("latest")
            baseline = comparing.get("baseline")
            result = res_dict[latest]  # result is dict
            expect = res_dict[baseline]
            tmp["待测事项"] = latest
            tmp["基线事项"] = baseline
            if comparing.get("precision") is not None:
                self.logger.get_log().info("{} 和 {} 精度(precision)对比验证开始".format(latest, baseline))
                # if result == "pass" or expect == "pass":
                if isinstance(result, str) or isinstance(expect, str):
                    if result == "pass" or expect == "pass":
                        self.logger.get_log().info("result: {} 和 expect: {}".format(result, expect))
                        tmp["precision"] = "ignore"
                        compare_res_list.append(tmp)
                        self.logger.get_log().info("{} 和 {} 豁免对比测试---".format(latest, baseline))
                    else:
                        exc += 1
                        self.logger.get_log().warn("{} 和 {} 标记为失败的对比测试---".format(latest, baseline))
                        tmp["precision"] = "failed"
                        compare_res_list.append(tmp)
                else:
                    # try:
                    #     precision = comparing.get("precision")
                    #     base_compare(
                    #         result=result,
                    #         expect=expect,
                    #         res_name=latest,
                    #         exp_name=baseline,
                    #         logger=self.logger.get_log(),
                    #         delta=precision.get("delta"),
                    #         rtol=precision.get("rtol"),
                    #     )
                    #     tmp["precision"] = "passed"
                    #     compare_res_list.append(tmp)
                    #     self.logger.get_log().info("{} 和 {} 精度对比通过~~~".format(latest, baseline))
                    # except Exception:
                    #     exc += 1
                    #     bug_trace = traceback.format_exc()
                    #     self.logger.get_log().warn("精度对比异常结果: {}".format(bug_trace))
                    #     tmp["precision"] = "failed"
                    #     compare_res_list.append(tmp)

                    precision = comparing.get("precision")
                    compare_res = base_compare(
                        result=result,
                        expect=expect,
                        res_name=latest,
                        exp_name=baseline,
                        logger=self.logger.get_log(),
                        delta=precision.get("delta"),
                        rtol=precision.get("rtol"),
                    )

                    if not compare_res:
                        tmp["precision"] = "passed"
                        compare_res_list.append(tmp)
                        self.logger.get_log().info("{} 和 {} 精度对比通过~~~".format(latest, baseline))
                    else:
                        exc += 1
                        tmp["precision"] = "failed"
                        compare_res_list.append(tmp)
                        self.logger.get_log().warn("{} 和 {} 精度对比失败！！".format(latest, baseline))

        self.logger.get_log().info("用例 {} 多执行器输出对比最终结果: {}".format(self.title, compare_res_list))
        if exc > 0:
            print("layer测试失败项目汇总: {}".format(compare_res_list))
            raise Exception("用例 {} 测试未通过".format(self.title))
            # assert False

    def _perf_case_run(self):
        """
        用于单个子图性能测试
        """
        exc = 0
        res_dict = {}
        # compare_res_list = []
        self.logger.get_log().info("测试case名称: {}".format(self.title))
        for testing in self.testings_list:
            try:
                self.logger.get_log().info("性能测试执行器: {}".format(testing))
                res = self._single_run(testing=testing, layerfile=self.layerfile)
                res_dict[testing] = res
            except Exception:
                bug_trace = traceback.format_exc()
                exc += 1
                res_dict[testing] = bug_trace
                self.logger.get_log().warn("性能执行器异常结果: {}".format(bug_trace))

        self.logger.get_log().info("用例 {} 多执行器性能结果: {}".format(self.title, res_dict))
        return res_dict, exc


if __name__ == "__main__":
    layerfile = "./layercase/sublayer1000/Det_cases/ppyolo_ppyolov2_r50vd_dcn_365e_coco/SIR_76.py"
    testing = "yaml/dy^dy2stcinn_eval_benchmark.yml"
    single_test = LayerTest(title="lzy_naive", layerfile=layerfile, testing=testing)
    single_test._perf_case_run()
