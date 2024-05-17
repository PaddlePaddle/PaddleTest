#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
测试执行器
"""
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import platform
from datetime import datetime
import layertest
from db.layer_db import LayerBenchmarkDB
from strategy.compare import perf_compare_dict
from tools.case_select import CaseSelect
from tools.logger import Logger
from tools.yaml_loader import YamlLoader
from tools.json_loader import JSONLoader
from tools.res_save import xlsx_save, download_sth, create_tar_gz, extract_tar_gz
from tools.upload_bos import UploadBos
from tools.statistics import split_list
from tools.alarm import Alarm


class Run(object):
    """
    最终执行接口
    """

    def __init__(self):
        """
        init
        """
        # 获取所有layer.yml文件路径
        self.layer_type = os.environ.get("CASE_TYPE")
        self.layer_dir = [os.path.join(self.layer_type, item) for item in os.environ.get("CASE_DIR").split(",")]

        # 获取需要忽略的case
        self.ignore_list = YamlLoader(yml=os.path.join("yaml", "ignore_case.yml")).yml.get(os.environ.get("TESTING"))

        # 获取测试集
        # self.py_list = CaseSelect(self.layer_dir, self.ignore_list).get_py_list(base_path=self.layer_dir)
        py_list = []
        for layer_dir in self.layer_dir:
            py_list = py_list + CaseSelect(layer_dir, self.ignore_list).get_py_list(base_path=layer_dir)

        # 测试集去重
        self.py_list = []
        for item in py_list:
            if not item in self.py_list:
                self.py_list.append(item)

        self.testing = os.environ.get("TESTING")
        self.py_cmd = os.environ.get("python_ver")
        self.report_dir = os.path.join(os.getcwd(), "report")

        self.logger = Logger("PaddleLTRun")
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 下载ground truth用于跨硬件测试
        plt_gt_download_url = os.environ.get("PLT_GT_DOWNLOAD_URL")
        if not plt_gt_download_url == "None" and os.environ.get("TESTING_MODE") == "precision":
            self.logger.get_log().info(f"下载plt_gt的url为: {plt_gt_download_url}")
            plt_gt_device = plt_gt_download_url.split("/")[-1]
            # if not os.path.exists(os.path.join("plt_gt_baseline", plt_gt_device)):
            #     os.makedirs(os.path.join("plt_gt_baseline", plt_gt_device))
            for testing in YamlLoader(yml=self.testing).get_junior_name("testings"):
                if not os.path.exists(os.path.join("plt_gt_baseline", plt_gt_device, testing)):
                    os.makedirs(os.path.join("plt_gt_baseline", plt_gt_device, testing))
                for py_file in self.py_list:
                    case_name = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
                    self.logger.get_log().info(f"开始下载plt_gt: {case_name}")
                    gt_url = f"{plt_gt_download_url}/{testing}/{case_name}.tensor"
                    download_sth(
                        gt_url=gt_url,
                        output_path=os.path.join("plt_gt_baseline", plt_gt_device, testing, f"{case_name}.tensor"),
                    )

    def _exit_code_txt(self, error_count, error_list):
        """"""
        core_dumps_list = self._core_dumps_case_count(report_path=self.report_dir)
        if error_count != 0 or core_dumps_list:
            self.logger.get_log().warning("测试失败, 下面进行bug分类统计: ")
            self.logger.get_log().warning(f"报错为core dumps的子图有: {core_dumps_list}")
            self.logger.get_log().info(f"测试子图总数为: {len(self.py_list)}")
            self.logger.get_log().warning(f"报错子图总数为: {len(error_list)}")
            self.logger.get_log().warning(f"报错为core dumps的子图数量为: {len(core_dumps_list)}")
            self.logger.get_log().warning(f"报错不为core dumps的异常子图数量为: {len(error_list)-len(core_dumps_list)}")
            self.logger.get_log().warning("请注意, 由于程序崩溃, core dumps的子图不会出现在allure报告中")
            self.logger.get_log().info(
                "『测试子图总数』 = 『下方回调'total'个数(allure报告中case总数)』 + 『core dump崩溃子图数』, 请核对case数目以保证测试无遗漏"
            )
            os.system("echo 7 > exit_code.txt")
        else:
            self.logger.get_log().info(f"测试子图总数为: {len(self.py_list)}")
            self.logger.get_log().info("测试通过, 无报错子图-。-")
            os.system("echo 0 > exit_code.txt")

    def _db_interact(self, sublayer_dict, error_list):
        """Database interaction"""
        # 数据库交互
        if os.environ.get("PLT_BM_DB") == "insert":  # 存入数据, 作为基线或对比
            layer_db = LayerBenchmarkDB(storage="apibm_config.yml")
            if os.environ.get("PLT_BM_MODE") == "baseline":
                layer_db.baseline_insert(data_dict=sublayer_dict, error_list=error_list)
            elif os.environ.get("PLT_BM_MODE") == "latest_as_baseline":
                baseline_dict, baseline_layer_type = layer_db.get_baseline_dict()
                compare_dict = perf_compare_dict(
                    baseline_dict=baseline_dict,
                    data_dict=sublayer_dict,
                    error_list=error_list,
                    baseline_layer_type=baseline_layer_type,
                    latest_layer_type=self.layer_type,
                )
                xlsx_save(
                    sublayer_dict=compare_dict,
                    excel_file=os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx",
                )
                # 先比较, 再更新baseline
                layer_db.baseline_insert(data_dict=sublayer_dict, error_list=error_list)
            elif os.environ.get("PLT_BM_MODE") == "latest":
                layer_db.latest_insert(data_dict=sublayer_dict, error_list=error_list)
                baseline_dict, baseline_layer_type = layer_db.get_baseline_dict()
                compare_dict = perf_compare_dict(
                    baseline_dict=baseline_dict,
                    data_dict=sublayer_dict,
                    error_list=error_list,
                    baseline_layer_type=baseline_layer_type,
                    latest_layer_type=self.layer_type,
                )
                xlsx_save(
                    sublayer_dict=compare_dict,
                    excel_file=os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx",
                )
            else:
                raise Exception(
                    "unknown benchmark mode, PaddleLT benchmark only support "
                    "baseline mode, latest_as_baseline mode or latest mode"
                )
        elif os.environ.get("PLT_BM_DB") == "select":  # 不存数据, 仅对比并生成表格
            layer_db = LayerBenchmarkDB(storage="apibm_config.yml")
            baseline_dict, baseline_layer_type = layer_db.get_baseline_dict()
            # layer_db.compare_with_baseline(data_dict=sublayer_dict, error_list=error_list)
            compare_dict = perf_compare_dict(
                baseline_dict=baseline_dict,
                data_dict=sublayer_dict,
                error_list=error_list,
                baseline_layer_type=baseline_layer_type,
                latest_layer_type=self.layer_type,
            )
            xlsx_save(
                sublayer_dict=compare_dict,
                excel_file=os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx",
            )
        elif os.environ.get("PLT_BM_DB") == "non-db":  # 不加载数据库，仅生成表格
            xlsx_save(
                sublayer_dict=sublayer_dict,
                excel_file=os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx",
            )
        else:
            Exception("unknown benchmark datebase mode, only support insert, select or nonuse")

    def _gt_upload(self):
        """精度groundtruth上传"""
        upload_url = os.environ.get("PLT_GT_UPLOAD_URL")
        _upload = UploadBos()
        if not upload_url == "None":
            self.logger.get_log().info(f"上传plt_gt的路径为: {os.environ.get('PLT_GT_UPLOAD_URL')}")
            for device in os.listdir("plt_gt"):
                device_path = os.path.join("plt_gt", device)
                for testing in os.listdir(device_path):
                    testing_path = os.path.join(device_path, testing)
                    for tensor in os.listdir(testing_path):
                        _upload.upload_to_bos(
                            bos_path=os.path.join(upload_url, device, testing),
                            file_path=os.path.join(testing_path, tensor),
                        )

    def _perf_upload(self):
        """性能表格/图表/原始数据上传"""
        bos_path = "PaddleLT/PaddleLTBenchmark/{}/build_{}".format(
            os.environ.get("PLT_BM_DB"), self.AGILE_PIPELINE_BUILD_ID
        )
        excel_file = os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx"
        if os.path.exists(excel_file):
            UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path=excel_file)
            self.logger.get_log().info("表格下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, excel_file))
        os.system("tar -czf plot.tar *.png")
        UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path="plot.tar")
        self.logger.get_log().info("plot下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, "plot.tar"))
        os.system("tar -czf pickle.tar *.pickle")
        UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path="pickle.tar")
        self.logger.get_log().info("pickle下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, "pickle.tar"))

        if os.environ.get("PLT_BM_EMAIL") == "True":
            alarm = Alarm(storage="apibm_config.yml")
            alarm.email_send(
                alarm.receiver,
                f"Paddle {self.layer_type}子图性能数据",
                f"表格下载链接: https://paddle-qa.bj.bcebos.com/{bos_path}/{excel_file}\n"
                f"plot下载链接: https://paddle-qa.bj.bcebos.com/{bos_path}/plot.tar\n"
                f"pickle下载链接: https://paddle-qa.bj.bcebos.com/{bos_path}/pickle.tar",
            )

    def _single_pytest_run(self, py_file):
        """run one test"""
        title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
        self.logger.get_log().info(f"开始测试子图 {title}, 准备执行pytest命令~~")
        if self.layer_type == "layerE2Ecase":
            exit_code = os.system(f"{self.py_cmd} -m pytest {py_file} --alluredir={self.report_dir}")
        else:
            exit_code = os.system(
                "cp -r PaddleLT.py {}.py && "
                "{} -m pytest {}.py --title={} --layerfile={} --testing={} --alluredir={}".format(
                    title, self.py_cmd, title, title, py_file, self.testing, self.report_dir
                )
            )
        self.logger.get_log().info(f"完成测试子图 {title}, 完成执行pytest命令~~")
        if exit_code != 0:
            return py_file, exit_code
        return None, None

    def _multithread_test_run(self):
        """multithread run some test"""
        error_list = []
        error_count = 0

        with ThreadPoolExecutor(max_workers=int(os.environ.get("MULTI_WORKER", 13))) as executor:
            # 提交任务给线程池
            futures = [executor.submit(self._single_pytest_run, py_file) for py_file in self.py_list]

            # 等待任务完成，并收集返回值
            for future in futures:
                _py_file, _exit_code = future.result()
                if _exit_code is not None:
                    error_list.append(_py_file)
                    error_count += 1

        self._gt_upload()
        self._exit_code_txt(error_count=error_count, error_list=error_list)

    def _test_run(self):
        """run some test"""
        error_list = []
        error_count = 0
        for py_file in self.py_list:
            _py_file, _exit_code = self._single_pytest_run(py_file=py_file)
            if _exit_code is not None:
                error_list.append(_py_file)
                error_count += 1

        self._gt_upload()
        self._exit_code_txt(error_count=error_count, error_list=error_list)

    def _multiprocess_perf_test_run(self):
        """
        multiprocess run some test
        需要处理cpu绑核以及gpu并行
        """

        def _queue_run(py_list, device_id, result_queue):
            """
            multi run main
            """
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PLT_DEVICE_ID") + str(device_id)
            sublayer_dict = {}
            error_count = 0
            error_list = []
            for py_file in py_list:
                title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
                single_test = layertest.LayerTest(title=title, layerfile=py_file, testing=self.testing)
                perf_dict, exit_code = single_test._perf_case_run()

                # 报错的子图+engine将不会收录进sublayer_dict
                if exit_code != 0:
                    error_list.append(py_file)
                    error_count += 1
                    continue

                sublayer_dict[title] = perf_dict

            # error_dict = self._run_main(all_cases=all_cases, loops=loops, base_times=base_times)

            result_queue.put(sublayer_dict, error_list, error_count)

        multiprocess_cases = split_list(lst=self.py_list, n=int(os.environ.get("MULTI_WORKER")))
        processes = []
        result_queue = multiprocessing.Queue()

        for i, cases_list in enumerate(multiprocess_cases):
            process = multiprocessing.Process(target=_queue_run, args=(cases_list, i, result_queue))
            process.start()
            # os.sched_setaffinity(process.pid, {self.core_index + i})
            processes.append(process)

        for process in processes:
            process.join()

        sublayer_dict = {}
        error_list = []
        error_count = 0
        while not result_queue.empty():
            single_sublayer_dict, single_error_list, single_error_count = result_queue.get()
            sublayer_dict.update(single_sublayer_dict)
            error_list.extend(single_error_list)
            error_count += single_error_count

        self._exit_code_txt(error_count=error_count, error_list=error_list)

        self._db_interact(sublayer_dict=sublayer_dict, error_list=error_list)
        self._perf_upload()

    def _perf_test_run(self):
        """run some test"""
        sublayer_dict = {}
        error_count = 0
        error_list = []
        for py_file in self.py_list:
            title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            single_test = layertest.LayerTest(title=title, layerfile=py_file, testing=self.testing)
            perf_dict, exit_code = single_test._perf_case_run()

            # 报错的子图+engine将不会收录进sublayer_dict
            if exit_code != 0:
                error_list.append(py_file)
                error_count += 1
                continue

            sublayer_dict[title] = perf_dict

        self._exit_code_txt(error_count=error_count, error_list=error_list)

        self._db_interact(sublayer_dict=sublayer_dict, error_list=error_list)
        self._perf_upload()
        self._pts_callback(error_count)

    def _core_dumps_case_count(self, report_path):
        """
        统计allure报告中无法展示的, core dumps程序崩溃的case
        """
        # 性能测试无allure report, 直接返回空
        if not os.path.exists(report_path):
            return []

        allure_case_list = []
        for json_file in os.listdir(report_path):
            if json_file.endswith("-result.json"):
                # layerE2Ecase中allure报告需要抓取的关键字, 与其他子图不一样
                if self.layer_type == "layerE2Ecase":
                    layer_name = JSONLoader(os.path.join(report_path, json_file)).json_dict()["labels"][-1]["value"]
                    allure_case_list.append(layer_name.replace(".", "/") + ".py")
                else:
                    layer_name = JSONLoader(os.path.join(report_path, json_file)).json_dict()["name"]
                    allure_case_list.append(layer_name.replace("^", "/") + ".py")

        all_case_list = []
        # 将./layercase/sublayer1000/Det_cases/gfl_gflv2_r50_fpn_1x_coco/SIR_173.py
        # 转换为layercase^sublayer1000^Det_cases^gfl_gflv2_r50_fpn_1x_coco^SIR_173
        for py_file in self.py_list:
            all_case_list.append(py_file)

        # 如果allure报告中不包含某个case, 说明这个case出现了core dumps
        core_dumps_list = []
        for case in all_case_list:
            if case not in allure_case_list:
                core_dumps_list.append(case)

        return core_dumps_list

    def _pts_callback(self, error_count):
        """
        用于性能任务回调pts. 精度任务通过start.sh最后的命令回调
        """
        pts_id = os.environ.get("pts_id")
        bos_path = "PaddleLT/PaddleLTBenchmark/{}/build_{}".format(
            os.environ.get("PLT_BM_DB"), self.AGILE_PIPELINE_BUILD_ID
        )
        excel_file = os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx"
        report_url = f"https://paddle-qa.bj.bcebos.com/{bos_path}/{excel_file}"

        success_count = len(self.py_list) - error_count
        if error_count > 0:
            status = "失败"
            result = f"success:{success_count} fail:{error_count}"
        else:
            status = "成功"
            result = f"success:{success_count} fail:{error_count}"

        os.system(f"bash ./callback.sh '{pts_id}' '{status}' '{result}' '{report_url}'")


if __name__ == "__main__":
    tes = Run()
    if os.environ.get("TESTING_MODE") == "precision":
        if os.environ.get("MULTI_WORKER") == "0":
            tes._test_run()
        else:
            tes._multithread_test_run()
    elif os.environ.get("TESTING_MODE") == "performance":
        if os.environ.get("MULTI_WORKER") == "0":
            tes._perf_test_run()
        else:
            tes._multiprocess_perf_test_run()
    else:
        raise Exception("unknown testing mode, PaddleLayerTest only support test precision or performance")
