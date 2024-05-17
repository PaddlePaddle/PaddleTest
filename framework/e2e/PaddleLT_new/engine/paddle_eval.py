#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
eval 方法
"""
import os
import traceback
import numpy as np
import paddle
from engine.paddle_xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData

from tools.logger import Logger

from tools.res_save import save_tensor


class LayerEval(object):
    """
    构建Layer评估的通用类
    """

    # def __init__(self, testing, layerfile, device_id):
    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.device = os.environ.get("PLT_SET_DEVICE")
        paddle.set_device(str(self.device))
        # paddle.set_device("{}:{}".format(str(self.device), str(device_id)))

        self.testing = testing
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.layerfile = layerfile
        # self.data = BuildData(layerfile=self.layerfile).get_single_data()

        self.use_multispec = os.environ.get("PLT_SPEC_USE_MULTI")

    def _net_input(self):
        """get input"""
        reset(self.seed)
        data = BuildData(layerfile=self.layerfile).get_single_data()
        return data

    def _net_instant(self):
        """get net and data"""
        reset(self.seed)
        net = BuildLayer(layerfile=self.layerfile).get_layer()
        return net

    def _net_input_and_spec(self):
        """get inputspec"""
        reset(self.seed)
        data, input_spec = BuildData(layerfile=self.layerfile).get_single_input_and_spec()
        return data, input_spec

    def _net_input_and_static_spec(self):
        """get static inputspec"""
        reset(self.seed)
        data, input_spec = BuildData(layerfile=self.layerfile).get_single_input_and_static_spec()
        return data, input_spec

    def _net_input_and_multi_spec(self):
        """get multi inputspec"""
        reset(self.seed)
        data, spec_gen = BuildData(layerfile=self.layerfile).get_single_input_and_multi_spec()
        return data, spec_gen

    # def _net_input_and_multi_spec_legacy(self):
    #     """get multi inputspec"""
    #     reset(self.seed)
    #     data, multi_input_spec = BuildData(layerfile=self.layerfile).get_single_input_and_multi_spec_legacy()
    #     return data, multi_input_spec

    def dy_eval(self):
        """dygraph eval"""
        net = self._net_instant()
        # net.eval()
        logit = net(*self._net_input())
        return {"logit": logit}

    def dy2st_eval(self):
        """dy2st eval"""
        net = self._net_instant()
        st_net = paddle.jit.to_static(net, full_graph=True)
        # net.eval()
        logit = st_net(*self._net_input())
        return {"logit": logit}

    def dy2st_eval_cinn(self):
        """dy2st eval"""
        net = self._net_instant()

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
        # net.eval()
        logit = cinn_net(*self._net_input())
        return {"logit": logit}

    def dy2st_eval_cinn_inputspec(self):
        """dy2st eval"""
        net = self._net_instant()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True

        if self.use_multispec == "True":
            # 如果不使用动态InputSpec都会报错, 则直接抛出异常跳过后续测试
            data, input_spec = self._net_input_and_static_spec()
            Logger("dy2st_eval_cinn_inputspec").get_log().info(f"待测动态InputSpec为: {input_spec}")
            cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True, input_spec=input_spec)
            # net.eval()
            logit = cinn_net(*data)

            # 开始测试动态InputSpec
            data, spec_gen = self._net_input_and_multi_spec()
            loops = (16**9) * 2
            i = 0
            # for i in range(loops):
            for inputspec in spec_gen.next():
                try:
                    Logger("dy2st_eval_cinn_inputspec").get_log().info(f"待测动态InputSpec为: {inputspec}")
                    cinn_net = paddle.jit.to_static(
                        net, build_strategy=build_strategy, full_graph=True, input_spec=inputspec
                    )
                    logit = cinn_net(*data)
                    if os.environ.get("PLT_SAVE_SPEC") == "True":
                        case_name = self.layerfile.replace(".py", "").replace("/", "^").replace(".", "^")
                        save_tensor(data=inputspec, filename=os.path.join("inputspec_save", f"{case_name}.inputspec"))
                    return {"logit": logit}
                except Exception:
                    bug_trace = traceback.format_exc()
                i += 1
                if i > loops:
                    break
            Logger("dy2st_eval_cinn_inputspec").get_log().warn(f"经过{loops}轮迭代测试, 动态InputSpec测试均失败。")
            raise Exception(bug_trace)
        else:
            data, input_spec = self._net_input_and_spec()
            cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True, input_spec=input_spec)
            # net.eval()
            logit = cinn_net(*data)
            return {"logit": logit}

    # def dy2st_eval_cinn_inputspec_legacy(self):
    #     """dy2st eval"""
    #     net = self._net_instant()
    #     build_strategy = paddle.static.BuildStrategy()
    #     build_strategy.build_cinn_pass = True

    #     if self.use_multispec == "True":
    #         data, multi_input_spec = self._net_input_and_multi_spec_legacy()
    #         multi_result = []
    #         for i, input_spec in enumerate(multi_input_spec):
    #             cinn_net = paddle.jit.to_static(
    #                 net, build_strategy=build_strategy, full_graph=True, input_spec=input_spec
    #             )
    #             logit = cinn_net(*data)
    #             multi_result.append({"logit": logit})
    #         return {"multi_result": multi_result}
    #     else:
    #         data, input_spec = self._net_input_and_spec()
    #         cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy,
    #                                         full_graph=True, input_spec=input_spec)
    #         # net.eval()
    #         logit = cinn_net(*data)
    #         return {"logit": logit}
