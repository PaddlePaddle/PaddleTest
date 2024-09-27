#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
train 方法
"""
import os
import numpy as np
import paddle
from engine.paddle_xtools import reset

from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData
from generator.builder_optimizer import BuildOptimizer
from generator.builder_loss import BuildLoss

from pltools.logger import Logger


class LayerTrain(object):
    """
    构建Layer训练的通用类
    """

    # def __init__(self, testing, layerfile, device_id):
    def __init__(self, testing, layerfile, device_place_id, upstream_net):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)
        self.device = os.environ.get("PLT_SET_DEVICE")
        paddle.device.set_device(f"{self.device}:{device_place_id}")
        Logger("LayerTrain.__init__").get_log().info(f"device_place_id is: {device_place_id}")
        # paddle.set_device("{}:{}".format(str(self.device), str(device_id)))

        self.testing = testing
        self.upstream_net = upstream_net
        self.return_net_instance = self.testing.get("return_net_instance", "False")
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.layerfile = layerfile
        self.step = self.testing.get("step")

    def _net_input(self):
        """get input"""
        reset(self.seed)
        data = BuildData(layerfile=self.layerfile).get_single_data()
        return data

    def _net_instant(self):
        """get net"""
        reset(self.seed)
        if self.upstream_net:
            net = self.upstream_net
        else:
            net = BuildLayer(layerfile=self.layerfile).get_layer()
        return net

    def _net_optimizer(self):
        """get optimizer"""
        reset(self.seed)
        optimizer_name = self.testing.get("optimizer").get("optimizer_name")
        optimizer_param = self.testing.get("optimizer").get("params")
        optimizer = BuildOptimizer(optimizer_name=optimizer_name, optimizer_param=optimizer_param)
        return optimizer

    def _net_loss(self):
        """get net"""
        reset(self.seed)
        loss_name = self.testing.get("Loss").get("loss_name")
        loss_param = self.testing.get("Loss").get("params")
        loss = BuildLoss(loss_name=loss_name, loss_param=loss_param)
        return loss

    def _net_input_and_spec(self):
        """get input and inputspec"""
        reset(self.seed)
        data, input_spec = BuildData(layerfile=self.layerfile).get_single_input_and_spec()
        return data, input_spec

    def _net_input_and_static_spec(self):
        """get input and static inputspec"""
        reset(self.seed)
        data, input_spec = BuildData(layerfile=self.layerfile).get_single_input_and_static_spec()
        return data, input_spec

    def _net_input_and_multi_spec(self):
        """get input and multi inputspec"""
        reset(self.seed)
        data, spec_gen = BuildData(layerfile=self.layerfile).get_single_input_and_multi_spec()
        return data, spec_gen

    # def _get_instant(self):
    #     """get data, net, optimizer, loss"""
    #     reset(self.seed)

    #     data = BuildData(layerfile=self.layerfile).get_single_data()
    #     net = BuildLayer(layerfile=self.layerfile).get_layer()

    #     optimizer_name = self.testing.get("optimizer").get("optimizer_name")
    #     optimizer_param = self.testing.get("optimizer").get("params")
    #     optimizer = BuildOptimizer(optimizer_name=optimizer_name, optimizer_param=optimizer_param)

    #     loss_name = self.testing.get("Loss").get("loss_name")
    #     loss_param = self.testing.get("Loss").get("params")
    #     loss = BuildLoss(loss_name=loss_name, loss_param=loss_param)
    #     return data, net, optimizer, loss

    def _get_data_grad(self, data):
        """记录list[inputs...]中的input.grad并生成list[input.grad...]"""
        data_grad = []
        for i in data:
            data_grad.append(i.grad)
        return data_grad

    def dy_train(self):
        """dygraph train"""
        # data, net, optimizer, loss = self._get_instant()
        data = self._net_input()
        net = self._net_instant()
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()
        # print(self.net.parameters()) 打印参数parameters

        # 构建optimizer用于训练
        if net.parameters():
            opt = optimizer.get_opt(net=net)

        for epoch in range(self.step):
            logit = net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy_train").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}

    def dy_dp_train(self):
        """dygraph data parallel train"""
        from paddle.distributed import fleet

        fleet.init(is_collective=True)

        data = self._net_input()
        net = self._net_instant()
        dp_net = fleet.distributed_model(net)
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()

        # 构建optimizer用于训练
        if net.parameters():
            opt = optimizer.get_opt(net=net)
            opt = fleet.distributed_optimizer(opt)

        for epoch in range(self.step):
            logit = dp_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if dp_net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy_dp_train").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        # return {"logit": logit, "data_grad": data_grad}
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}

    # def dy_train_dl(self):
    #     """dygraph train with dataloader"""
    #     reset(self.seed)

    #     # net = self.net.get_layer()
    #     self.net.train()

    #     # 构建optimizer用于训练
    #     opt = self.optimizer.get_opt(net=self.net)

    #     for epoch in range(self.step):
    #         for i, data_dict in enumerate(self.data()):
    #             logit = self.net(**data_dict)
    #             # 构建loss用于训练
    #             # logit = self.loss_info.get_loss(logit)
    #             loss = self.loss.get_loss(logit)
    #             loss.backward()
    #             opt.step()
    #             opt.clear_grad()
    #     return logit

    def dy2st_train(self):
        """dy2st train"""

        # if not self.net.parameters():
        #     return "pass"

        data = self._net_input()
        net = self._net_instant()
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()
        st_net = paddle.jit.to_static(net, full_graph=True)

        # 构建optimizer用于训练
        if st_net.parameters():
            opt = optimizer.get_opt(net=st_net)

        for epoch in range(self.step):
            logit = st_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if st_net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy2st_train").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        # return {"logit": logit, "data_grad": data_grad}
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": st_net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}

    def dy2st_train_inputspec(self):
        """dy2st cinn train with inputspec"""
        data, input_spec = self._net_input_and_spec()
        Logger("dy2st_train_inputspec").get_log().info(f"待测动态InputSpec为: {input_spec}")
        net = self._net_instant()
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()
        st_net = paddle.jit.to_static(net, full_graph=True, input_spec=input_spec)

        # 构建optimizer用于训练
        if st_net.parameters():
            opt = optimizer.get_opt(net=st_net)

        for epoch in range(self.step):
            logit = st_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if st_net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy2st_train_inputspec").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        # return {"logit": logit, "data_grad": data_grad}
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": st_net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}

    def dy2st_train_static_inputspec(self):
        """dy2st cinn train with inputspec"""
        data, input_spec = self._net_input_and_static_spec()
        Logger("dy2st_train_static_inputspec").get_log().info(f"待测静态InputSpec为: {input_spec}")
        net = self._net_instant()
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()
        st_net = paddle.jit.to_static(net, full_graph=True, input_spec=input_spec)

        # 构建optimizer用于训练
        if st_net.parameters():
            opt = optimizer.get_opt(net=st_net)

        for epoch in range(self.step):
            logit = st_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if st_net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy2st_train_static_inputspec").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        # return {"logit": logit, "data_grad": data_grad}
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": st_net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}

    def dy2st_train_cinn(self):
        """dy2st cinn train"""
        data = self._net_input()
        net = self._net_instant()
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)

        # 构建optimizer用于训练
        if cinn_net.parameters():
            opt = optimizer.get_opt(net=cinn_net)

        for epoch in range(self.step):
            logit = cinn_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if cinn_net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy2st_train_cinn").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        # return {"logit": logit, "data_grad": data_grad}
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": cinn_net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}

    def dy2st_train_cinn_inputspec(self):
        """dy2st cinn train with inputspec"""
        data, input_spec = self._net_input_and_spec()
        Logger("dy2st_train_cinn_inputspec").get_log().info(f"待测动态InputSpec为: {input_spec}")
        net = self._net_instant()
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True, input_spec=input_spec)

        # 构建optimizer用于训练
        if cinn_net.parameters():
            opt = optimizer.get_opt(net=cinn_net)

        for epoch in range(self.step):
            logit = cinn_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if cinn_net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy2st_train_cinn_inputspec").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        # return {"logit": logit, "data_grad": data_grad}
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": cinn_net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}

    def dy2st_train_cinn_static_inputspec(self):
        """dy2st cinn train with inputspec"""
        data, input_spec = self._net_input_and_static_spec()
        Logger("dy2st_train_cinn_static_inputspec").get_log().info(f"待测静态InputSpec为: {input_spec}")
        net = self._net_instant()
        optimizer = self._net_optimizer()
        loss = self._net_loss()

        net.train()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True, input_spec=input_spec)

        # 构建optimizer用于训练
        if cinn_net.parameters():
            opt = optimizer.get_opt(net=cinn_net)

        for epoch in range(self.step):
            logit = cinn_net(*data)
            # 构建loss用于训练
            dy_loss = loss.get_loss(logit)
            dy_loss.backward()
            if cinn_net.parameters():
                opt.step()
                opt.clear_grad()

        Logger("dy2st_train_cinn_static_inputspec").get_log().info(f"已完成 {epoch} 轮训练")
        data_grad = self._get_data_grad(data)
        # return {"logit": logit, "data_grad": data_grad}
        if self.return_net_instance == "True":
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": cinn_net}
        else:
            return {"res": {"logit": logit, "data_grad": data_grad}, "net": None}
