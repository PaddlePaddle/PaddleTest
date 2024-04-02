#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
det
"""
import copy
import traceback
import numpy as np
import paddle
import ppdet

paddle.seed(33)
np.random.seed(33)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


def run_test(to_static, to_train):
    """train"""
    paddle.enable_static()
    paddle.disable_static()
    paddle.seed(33)
    np.random.seed(33)
    # paddle.set_default_dtype("float32")
    # input = {'image': paddle.to_tensor(randtool("float", -1, 1, shape=[4, 3, 224, 224]).astype("float64"))}
    input = {
        "out_transformer": (
            paddle.to_tensor(randtool("float", -1, 1, shape=[6, 2, 100, 256]).astype("float32")),
            paddle.to_tensor(randtool("float", -1, 1, shape=[2, 256, 24, 32]).astype("float32")),
            paddle.to_tensor(randtool("float", -1, 1, shape=[2, 256, 24, 32]).astype("float32")),
            paddle.to_tensor(randtool("float", -1, 1, shape=[2, 1, 1, 24, 32]).astype("float32")),
        ),
        "body_feats": [paddle.to_tensor(randtool("float", 0, 1, shape=[2, 2048, 24, 32]).astype("float32"))],
        "inputs": {
            "im_id": paddle.to_tensor([[424481], [277746]], dtype="int64"),
            "is_crowd": [
                paddle.to_tensor([[0], [0]], dtype="int32"),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype="int32"),
            ],
            "gt_class": [
                paddle.to_tensor([[4], [0]], dtype="int32"),
                paddle.to_tensor([[0], [71], [71], [26], [41], [71], [71], [71], [71]], dtype="int32"),
            ],
            "gt_bbox": [
                paddle.to_tensor(randtool("float", 0, 1, shape=[2, 4]).astype("float32")),
                paddle.to_tensor(randtool("float", 0, 1, shape=[9, 4]).astype("float32")),
            ],
            "curr_iter": paddle.to_tensor([[0], [1]], dtype="int64"),
            "image": paddle.to_tensor(randtool("float", -1, 1, shape=[2, 3, 768, 1024]).astype("float32")),
            "im_shape": paddle.to_tensor([[576.0, 863.0], [768.0, 1024.0]], dtype="float32"),
            "scale_factor": paddle.to_tensor([[1.34894609, 1.34843755], [1.60000002, 1.60000002]], dtype="float32"),
            "pad_mask": paddle.to_tensor(randtool("float", 0, 1, shape=[2, 768, 1024]).astype("float32")),
            "epoch_id": 0,
        },
    }
    net = ppdet.modeling.heads.detr_head.DETRHead(
        num_classes=80,
        hidden_dim=256,
        nhead=8,
        num_mlp_layers=3,
        loss=ppdet.modeling.losses.DETRLoss(matcher=ppdet.modeling.transformers.matchers.HungarianMatcher()),
        fpn_dims=[],
        with_mask_head=False,
        use_focal_loss=False,
    )
    if to_static:
        net = paddle.jit.to_static(net)

    # print("net parameters is: ", net.parameters())

    opt = paddle.optimizer.SGD(learning_rate=0.000001, parameters=net.parameters())
    # dygraph train
    if to_train is True:
        for epoch in range(3):
            logit = net(**input)
            logit = (
                logit["loss_bbox"]
                + logit["loss_giou"]
                + 0.1 * logit["loss_class_aux"]
                + 0.1 * logit["loss_bbox_aux"]
                + 0.1 * logit["loss_giou_aux"]
            )
            logit.backward()
            opt.step()
            opt.clear_grad()

        return logit
    else:
        net.eval()
        logit = net(**input)
        # print('eval logit is: ', logit)
        # logit = logit["loss_bbox"] + logit["loss_giou"] + 0.1 * logit["loss_class_aux"] + 0.1 * logit[
        #     "loss_bbox_aux"] + 0.1 * logit["loss_giou_aux"]
        return logit


try:
    dy_out_final = run_test(to_static=True, to_train=True)

except Exception:
    bug_trace = traceback.format_exc()
    print("dygraph to static train is Failed!!! ")
    print(bug_trace)

dy_out_final = run_test(to_static=False, to_train=True)

# st_out_final = train(True)
# 结果打印
print("dy_out_final", dy_out_final)
# print("st_out_final", st_out_final)
# print(np.array_equal(dy_out_final.numpy(), st_out_final.numpy()))

# print("diff is: ", dy_out_final - st_out_final)
