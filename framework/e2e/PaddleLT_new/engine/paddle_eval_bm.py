#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
eval 方法
"""
import os
import timeit
import numpy as np
import paddle
from engine.paddle_xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData
from tools.res_save import save_pickle
from tools.statistics import trimmean, mean, best, best_top_k, perf_by_step
from tools.logger import Logger


class LayerEvalBM(object):
    """
    构建Layer评估的性能通用类
    """

    # def __init__(self, testing, layerfile, device_id):
    def __init__(self, testing, layerfile, device_place_id):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.device = os.environ.get("PLT_SET_DEVICE")
        # paddle.set_device(str(self.device))
        paddle.set_device(f"{self.device}:{device_place_id}")
        # paddle.set_device("{}:{}".format(str(self.device), str(device_id)))

        self.perf_repeat = int(os.environ.get("PLT_BM_REPEAT", "100"))
        self.perf_statis = os.environ.get("PLT_BM_STATIS", "trimmean")
        self.timeit_num = int(os.environ.get("TIMEIT_NUM", "1"))
        self.statis_times = 100
        self.statis_round = 6

        self.testing = testing
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.layerfile = layerfile
        self.data = BuildData(layerfile=self.layerfile).get_single_tensor()
        self.logger = Logger("LayerEvalBM")

    def _net_instant(self):
        """get net and data"""
        reset(self.seed)
        net = BuildLayer(layerfile=self.layerfile).get_layer()
        return net

    def _set_cinn_flags(self):
        """
        set cinn flags
        """
        os.environ["FLAGS_prim_all"] = "true"
        os.environ["FLAGS_prim_enable_dynamic"] = "true"

        os.environ["FLAGS_use_cinn"] = "true"
        os.environ["FLAGS_cinn_bucket_compile"] = "1"
        os.environ["FLAGS_cinn_new_group_scheduler"] = "1"
        os.environ["FLAGS_group_schedule_tiling_first"] = "1"

        os.environ["FLAGS_enable_pir_api"] = "1"
        os.environ["FLAGS_enable_pir_in_executor"] = "1"
        os.environ["MIN_GRAPH_SIZE"] = "0"

        self.logger.get_log().info("_set_cinn_flags 性能测试过程中, 成功追加设定cinn相关FLAGS~~")

    def _del_cinn_flags(self):
        """
        del cinn flags
        """
        if "FLAGS_prim_all" in os.environ:
            del os.environ["FLAGS_prim_all"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_prim_all~~")
        if "FLAGS_prim_enable_dynamic" in os.environ:
            del os.environ["FLAGS_prim_enable_dynamic"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_prim_enable_dynamic~~")

        if "FLAGS_use_cinn" in os.environ:
            del os.environ["FLAGS_use_cinn"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_use_cinn~~")
        if "FLAGS_cinn_bucket_compile" in os.environ:
            del os.environ["FLAGS_cinn_bucket_compile"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_cinn_bucket_compile~~")
        if "FLAGS_cinn_new_group_scheduler" in os.environ:
            del os.environ["FLAGS_cinn_new_group_scheduler"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_cinn_new_group_scheduler~~")
        if "FLAGS_group_schedule_tiling_first" in os.environ:
            del os.environ["FLAGS_group_schedule_tiling_first"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_group_schedule_tiling_first~~")

        if "FLAGS_enable_pir_api" in os.environ:
            del os.environ["FLAGS_enable_pir_api"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_enable_pir_api~~")
        if "FLAGS_enable_pir_in_executor" in os.environ:
            del os.environ["FLAGS_enable_pir_in_executor"]
            self.logger.get_log().info("_del_cinn_flags del FLAGS_enable_pir_in_executor~~")
        if "MIN_GRAPH_SIZE" in os.environ:
            del os.environ["MIN_GRAPH_SIZE"]
            self.logger.get_log().info("_del_cinn_flags del MIN_GRAPH_SIZE~~")

        self.logger.get_log().info("_del_cinn_flags 性能测试过程中, 取消cinn相关FLAGS~~")

    def dy_eval_perf(self):
        """dygraph eval"""
        self._del_cinn_flags()

        net = self._net_instant()
        net.eval()

        def _perf(input_data):
            logit = net(*input_data)
            return logit

        total_time_list = []
        # 预热
        timeit.timeit(lambda: _perf(self.data), number=10)
        # timeit.timeit(lambda: _perf(self.data), number=int(self.perf_repeat * self.timeit_num * 0.2))
        for i in range(self.perf_repeat):
            total_time = timeit.timeit(lambda: _perf(self.data), number=self.timeit_num)
            paddle.core._cuda_synchronize(paddle.CUDAPlace(0))
            total_time_list.append(total_time)

        save_pickle(data=total_time_list, filename="dy_eval_perf_" + self.layerfile)
        # 画图
        perf_by_step(
            data_list=total_time_list,
            step_scale=[0.1, 0.5, 1],
            filename="dy_eval_perf_" + self.layerfile + "_by_step",
        )

        time_res = eval(self.perf_statis)(data_list=total_time_list)
        time_res = round(time_res * self.statis_times, self.statis_round)

        self._set_cinn_flags()
        return time_res

    def dy2st_eval_perf(self):
        """dygraph eval"""
        self._del_cinn_flags()

        net = self._net_instant()
        st_net = paddle.jit.to_static(net, full_graph=True)
        st_net.eval()

        def _perf(input_data):
            logit = st_net(*input_data)
            return logit

        total_time_list = []
        # 预热
        timeit.timeit(lambda: _perf(self.data), number=10)
        # timeit.timeit(lambda: _perf(self.data), number=int(self.perf_repeat * self.timeit_num * 0.2))
        for i in range(self.perf_repeat):
            total_time = timeit.timeit(lambda: _perf(self.data), number=self.timeit_num)
            total_time_list.append(total_time)

        save_pickle(data=total_time_list, filename="dy2st_eval_perf_" + self.layerfile)
        # 画图
        perf_by_step(
            data_list=total_time_list,
            step_scale=[0.1, 0.5, 1],
            filename="dy2st_eval_perf_" + self.layerfile + "_by_step",
        )

        time_res = eval(self.perf_statis)(data_list=total_time_list)
        time_res = round(time_res * self.statis_times, self.statis_round)

        self._set_cinn_flags()
        return time_res

    def dy2st_eval_cinn_perf(self):
        """dy2st eval"""
        self._set_cinn_flags()

        net = self._net_instant()

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
        cinn_net.eval()

        # net.eval()
        def _perf(input_data):
            logit = cinn_net(*input_data)
            return logit

        total_time_list = []
        # 预热
        timeit.timeit(lambda: _perf(self.data), number=10)
        # timeit.timeit(lambda: _perf(self.data), number=int(self.perf_repeat * self.timeit_num * 0.2))
        for i in range(self.perf_repeat):
            total_time = timeit.timeit(lambda: _perf(self.data), number=self.timeit_num)
            total_time_list.append(total_time)

        save_pickle(data=total_time_list, filename="dy2st_eval_cinn_perf_" + self.layerfile + "_total_time_list")
        # 画图
        perf_by_step(
            data_list=total_time_list,
            step_scale=[0.1, 0.5, 1],
            filename="dy2st_eval_cinn_perf_" + self.layerfile + "_by_step",
        )

        time_res = eval(self.perf_statis)(data_list=total_time_list)
        time_res = round(time_res * self.statis_times, self.statis_round)

        self._set_cinn_flags()
        return time_res
