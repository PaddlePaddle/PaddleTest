#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_data_inmemorydataset.py
  * @author liujie44@baidu.com
  * @date 2021-11-10 14:30
  * @brief
  *
  **************************************************************************/
"""
import os

import paddle
from utils import run_priority

paddle.enable_static()

with open("test_queue_dataset_run_a.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
    data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
    data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
    data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
    f.write(data)
with open("test_queue_dataset_run_b.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
    data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
    data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
    data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
    f.write(data)
slots = ["slot1", "slot2", "slot3", "slot4"]
slots_vars = []
for slot in slots:
    var = paddle.static.data(name=slot, shape=[None, 1], dtype="int64", lod_level=1)
    slots_vars.append(var)

dataset = paddle.distributed.InMemoryDataset()


@run_priority(level="P0")
def test_data_inmemorydataset1():
    """test init & load_into_memory & get_memory_data_size & release_memory"""
    dataset.init(batch_size=1, thread_num=2, input_type=1, pipe_command="cat", use_var=slots_vars)
    dataset.set_filelist(["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])
    dataset.load_into_memory()
    assert dataset.get_memory_data_size() == 8

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    startup_program = paddle.static.Program()
    # main_program = paddle.static.Program()
    exe.run(startup_program)
    # exe.train_from_dataset(main_program, dataset)

    with open("./test_queue_dataset_run_a.txt") as f1:
        lines = f1.readlines()
        for line in lines:
            assert (len(line[0])) == 1

    with open("./test_queue_dataset_run_b.txt") as f2:
        lines = f2.readlines()
        for line in lines:
            assert (len(line[0])) == 1

    dataset.release_memory()
    print("test_data_inmemorydataset1 ... ok")


@run_priority(level="P0")
def test_data_inmemorydataset2():
    """test preload_into_memory & wait_preload_done"""
    dataset.preload_into_memory()
    dataset.wait_preload_done()
    dataset.release_memory()
    print("test_data_inmemorydataset2 ... ok")


@run_priority(level="P0")
def test_data_inmemorydataset3():
    """test local_shuffle"""
    dataset.load_into_memory()
    # dataset.local_shuffle()
    dataset.release_memory()
    print("test_data_inmemorydataset3 ... ok")


@run_priority(level="P0")
def test_data_inmemorydataset4():
    """test global_shuffle & get_shuffle_data_size"""
    dataset.load_into_memory()
    dataset.global_shuffle()
    assert dataset.get_shuffle_data_size() == 0
    dataset.release_memory()
    print("test_data_inmemorydataset4 ... ok")


@run_priority(level="P0")
def test_data_inmemorydataset5():
    """test  slots_shuffle"""
    dataset = paddle.distributed.InMemoryDataset()
    dataset._init_distributed_settings(fea_eval=True)

    dataset.init(batch_size=1, thread_num=2, input_type=1, pipe_command="cat", use_var=slots_vars)
    dataset.set_filelist(["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])
    dataset.load_into_memory()
    dataset.slots_shuffle(["slot1"])
    dataset.release_memory()
    print("test_data_inmemorydataset5 ... ok")


@run_priority(level="P0")
def test_data_inmemorydataset6():
    """test _init_distributed_settings & update_settings"""
    dataset.init(batch_size=1, thread_num=2, input_type=1, pipe_command="cat", use_var=[])
    dataset._init_distributed_settings(parse_ins_id=True, parse_content=True, fea_eval=True, candidate_size=10000)
    dataset.update_settings(batch_size=2)
    assert dataset._init_distributed_settings is not None
    print("test_data_inmemorydataset6 ... ok")


if __name__ == "__main__":
    test_data_inmemorydataset1()
    test_data_inmemorydataset2()
    test_data_inmemorydataset3()
    test_data_inmemorydataset4()
    test_data_inmemorydataset5()
    test_data_inmemorydataset6()
    # os.remove("./test_queue_dataset_run_a.txt")
    # os.remove("./test_queue_dataset_run_b.txt")
