#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
测试执行器
"""
import os
import traceback

# from engine.engine_map import engine_map
from strategy.compare import base_compare
from tools.yaml_loader import YamlLoader
from tools.logger import Logger
from tools.res_save import save_tensor, load_tensor, save_pickle


class LayerTest(object):
    """
    单个Layer case 执行
    """

    def __init__(self, title, layerfile, testing, device_place_id=0):
        """ """
        self.title = title

        self.device_place_id = int(device_place_id)
        self.layerfile = layerfile.replace(".py", "").replace("/", ".").lstrip(".")

        # 解析testing.yml
        self.test_config = YamlLoader(yml=testing)
        self.testings = self.test_config.yml.get("testings")
        self.testings_list = self.test_config.get_junior_name("testings")

        self.compare_list = self.test_config.yml.get("compare")

        self.logger = Logger("PaddleLT")
        self.report_dir = os.path.join(os.getcwd(), "report")

    def _single_run(self, testing, layerfile, device_place_id=0):
        """
        单次执行器测试
        :param testing: 'dy_train', 'dy_eval'...
        :return:
        """
        if os.environ.get("FRAMEWORK") == "paddle":
            from engine.paddle_engine_map import paddle_engine_map as engine_map
        elif os.environ.get("FRAMEWORK") == "torch":
            from engine.torch_engine_map import torch_engine_map as engine_map
        layer_test = engine_map[testing](
            testing=self.testings.get(testing), layerfile=layerfile, device_place_id=device_place_id
        )
        res = getattr(layer_test, testing)()
        return res

    def _case_run(self):
        """
        用于单个子图精度测试
        """
        exc_func = 0
        exc = 0
        res_dict = {}
        compare_res_list = []
        self.logger.get_log().info("测试case名称: {}".format(self.title))
        fail_testing_list = []
        for testing in self.testings_list:
            try:
                self.logger.get_log().info("测试执行器: {}".format(testing))
                res = self._single_run(testing=testing, layerfile=self.layerfile, device_place_id=self.device_place_id)
                res_dict[testing] = res
                if os.environ.get("PLT_SAVE_GT") == "True":  # 开启gt保存
                    gt_path = os.path.join("plt_gt", os.environ.get("PLT_SET_DEVICE"), testing)
                    if not os.path.exists(gt_path):
                        os.makedirs(gt_path)
                    save_tensor(res, os.path.join(gt_path, self.title))
            except Exception:
                bug_trace = traceback.format_exc()
                exc_func += 1
                res_dict[testing] = bug_trace
                fail_testing_list.append(testing)
                self.logger.get_log().warn("执行器异常结果: {}".format(bug_trace))

        if exc_func > 0:
            self.logger.get_log().warn("layer测试失败项目汇总: {}".format(fail_testing_list))
            self.logger.get_log().warn("用例 {} 测试未通过".format(self.title))
            raise Exception(bug_trace)

        for comparing in self.compare_list:
            tmp = {}
            latest = comparing.get("latest")
            # if "baseline" not in comparing or comparing["baseline"] is None:
            if isinstance(comparing.get("baseline"), dict):  # 加载plt_gt_baseline中的真值作为基线
                baseline_info = comparing.get("baseline")
                # gt_dir = baseline_info.get("gt_dir")
                gt_dir = "plt_gt_baseline"
                gt_device = baseline_info.get("device")
                baseline = baseline_info.get("testing")
                gt_path = os.path.join(gt_dir, gt_device, baseline, self.title)
                expect = load_tensor(gt_path)
            else:  # 使用res_dict中的测试结果作为基线
                baseline = comparing.get("baseline")
                expect = res_dict[baseline]
            result = res_dict[latest]  # result is dict
            # expect = res_dict[baseline]
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
        if exc + exc_func > 0:
            self.logger.get_log().warn("layer精度对比异常汇总: {}".format(compare_res_list))
            # raise Exception("用例 {} 测试未通过".format(self.title))
            assert False

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

    def _perf_unit_case_run(self, plt_exc):
        """
        用于单个子图性能测试
        """
        exc = 0
        res_dict = {}
        # compare_res_list = []
        self.logger.get_log().info("测试case名称: {}".format(self.title))
        if plt_exc in self.testings_list:
            try:
                self.logger.get_log().info("性能测试执行器: {}".format(plt_exc))
                res = self._single_run(testing=plt_exc, layerfile=self.layerfile)
                res_dict[plt_exc] = res
            except Exception:
                bug_trace = traceback.format_exc()
                exc += 1
                res_dict[plt_exc] = bug_trace
                self.logger.get_log().warn("性能执行器异常结果: {}".format(bug_trace))

        self.logger.get_log().info("用例 {} 单执行器性能结果: {}".format(self.title, res_dict))
        if not os.path.exists("./perf_unit_result"):
            os.makedirs(name="./perf_unit_result")
        save_pickle(data=[res_dict, exc], filename=os.path.join("perf_unit_result", self.title + "-" + plt_exc))


if __name__ == "__main__":
    # # 精度调试逻辑
    # layerfile = "./layerTorchcase/demo/SIR_101.py"
    # testing = "yaml/dy_eval.yml"
    # single_test = LayerTest(title="lzy_naive", layerfile=layerfile, testing=testing)
    # single_test._case_run()

    # 性能调试逻辑
    if os.environ.get("TESTING_MODE") == "performance":
        if os.environ.get("PLT_PERF_MODE") == "unit-python":
            import argparse

            parser = argparse.ArgumentParser(description=__doc__)
            # 用于性能测试 单执行器+单子图的 独立python执行模式
            parser.add_argument("--layerfile", type=str, default="layercase/demo/SIR_101.py", help="子图路径")
            parser.add_argument("--testing", type=str, default="yaml/dy_eval.yml", help="执行器配置")
            parser.add_argument("--plt_exc", type=str, default="dy_eval", help="单执行器选择")
            args = parser.parse_args()
            py_file = args.layerfile
            title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            single_test = LayerTest(title=title, layerfile=args.layerfile, testing=args.testing)
            single_test._perf_unit_case_run(plt_exc=args.plt_exc)
    else:
        pass
