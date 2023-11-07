#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file dist_auto_comp_cost.py
  * @author liujie44@baidu.com
  * @date 2022-09-07 11:00
  * @brief
  *
  **************************************************************************/
"""
import json
import os

from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost.comp_op_cost import (
    AssignOpCost,
    AssignValueOpCost,
    BeamSearchDecodeOpCost,
    BeamSearchOpCost,
    CastOpCost,
    ConcatOpCost,
    DropoutGradOpCost,
    ElementwiseAddGradOpCost,
    ElementwiseAddOpCost,
    ElementwiseDivGradOpCost,
    ElementwiseDivOpCost,
    ElementwiseMulGradOpCost,
    ElementwiseMulOpCost,
    ElementwiseSubOpCost,
    EmbeddingGradOpCost,
    EmbeddingOpCost,
    FillConstantBatchSizeLikeOpCost,
    FillConstantOpCost,
    FusedSoftmaxMaskUpperTriangleGradOpCost,
    FusedSoftmaxMaskUpperTriangleOpCost,
    GatherOpCost,
    GeluGradOpCost,
    GeluOpCost,
    GreaterEqualOpCost,
    IncrementOpCost,
    IsEmptyOpCost,
    LayerNormGradOpCost,
    LayerNormOpCost,
    LessThanOpCost,
    LodResetOpCost,
    LogicalAndOpCost,
    LogicalNotOpCost,
    LogOpCost,
    LookupTableV2GradOpCost,
    LookupTableV2OpCost,
    MatmulOpCost,
    MatmulV2GradOpCost,
    MatmulV2OpCost,
    MemcpyOpCost,
    MulGradOpCost,
    MulOpCost,
    OneHotOpCost,
    ReadFromArrayOpCost,
    ReduceMeanGradOpCost,
    ReduceMeanOpCost,
    ReduceSumGradOpCost,
    ReduceSumOpCost,
    Reshape2GradOpCost,
    Reshape2OpCost,
    SamplingIdOpCost,
    ScaleOpCost,
    SliceOpCost,
    SoftmaxGradOpCost,
    SoftmaxOpCost,
    SoftmaxWithCrossEntropyGradOpCost,
    SoftmaxWithCrossEntropyOpCost,
    SplitOpCost,
    SquareGradOpCost,
    SquareOpCost,
    Squeeze2OpCost,
    SumOpCost,
    TopKOpCost,
    Transpose2GradOpCost,
    Transpose2OpCost,
    Unsqueeze2OpCost,
    WriteToArrayOpCost,
)

from dist_auto_cluster import cluster_json
from utils import run_priority


@run_priority(level="P0")
def test_comp_cost():
    """test_comp_cost"""
    # Build cluster
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
    cluster_json_object = json.loads(cluster_json)
    with open(cluster_json_path, "w") as cluster_json_file:
        json.dump(cluster_json_object, cluster_json_file)
    cluster = Cluster()
    cluster.build_from_file(cluster_json_path)

    op_cost = AssignOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = AssignValueOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = BeamSearchOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = BeamSearchDecodeOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = CastOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ConcatOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ElementwiseAddOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ElementwiseAddGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ElementwiseDivOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ElementwiseDivGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ElementwiseMulOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ElementwiseMulGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ElementwiseSubOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = EmbeddingOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = EmbeddingGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = FillConstantOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = FillConstantBatchSizeLikeOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = GatherOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = GeluOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = GeluGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = GreaterEqualOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = IncrementOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = IsEmptyOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LayerNormOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LayerNormGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LessThanOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LogicalNotOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LogicalAndOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LodResetOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LogOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LookupTableV2OpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = LookupTableV2GradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = MatmulOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = MatmulV2OpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = MatmulV2GradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = MemcpyOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = MulOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = MulGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = OneHotOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ReadFromArrayOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ReduceSumOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ReduceSumGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = Reshape2OpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = MatmulV2OpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = Reshape2GradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ReduceMeanOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ReduceMeanGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SamplingIdOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = ScaleOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SliceOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SoftmaxOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SoftmaxGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SoftmaxWithCrossEntropyOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SoftmaxWithCrossEntropyGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SplitOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = Squeeze2OpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SquareOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SquareGradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = SumOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = TopKOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = Transpose2OpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = Transpose2GradOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = Unsqueeze2OpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    op_cost = WriteToArrayOpCost(cluster=cluster)
    assert op_cost.flops >= 0
    assert op_cost.time >= 0
    assert op_cost.memory >= 0

    print("test_comp_cost ... ok")


if __name__ == "__main__":
    test_comp_cost()
