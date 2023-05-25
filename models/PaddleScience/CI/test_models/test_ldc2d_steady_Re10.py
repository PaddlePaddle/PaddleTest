# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入模块
import os
import sys
import subprocess

import pytest

from tools.log_analysis import get_last_epoch_loss, get_last_eval_metric


def test_ldc2d_steady_Re10_exit_code():
    """
    测试函数：测试 ldc2d_steady_Re10.py 脚本的退出码是否为 0 以保证可视化文件的正常保存
    """
    # 定义变量
    output_dir = "./ldc2d_steady_Re10"  # 输出目录
    epoch_num = 10  # 迭代次数
    py_version = os.getenv("py_version", "3.8")  # Python 版本号，从环境变量中获取，默认值为3.8

    # 执行命令行命令，运行 ldc2d_unsteady_Re10.py 脚本
    command = (
        f"python{py_version} ../../examples/ldc/ldc2d_steady_Re10.py --epochs={epoch_num} --output_dir={output_dir}"
    )
    process = subprocess.Popen(command, shell=True)

    # 等待脚本执行完成，并返回退出码
    exit_code = process.wait()

    # 断言退出码为 0
    assert exit_code == 0


def test_ldc2d_steady_Re10_loss():
    """
    test loss
    """
    epoch_num = 10  # 迭代次数
    output_dir = "./ldc2d_steady_Re10"  # 输出目录
    base_loss = 56.74955  # 基准损失值

    # 获取训练过程的日志文件并计算最后一轮迭代的损失值
    log_file = os.path.join(output_dir, "train.log")
    last_loss = get_last_epoch_loss(log_file, epoch_num)

    # 断言最后一轮迭代的损失值与基准
    assert float(last_loss) == base_loss


def test_ldc2d_steady_Re10_metric():
    """
    测试函数：测试 ldc2d_steady_Re10.py 的评估值
    """
    output_dir = "./ldc2d_steady_Re10"  # 输出目录
    loss_function = "Residual"  # 损失函数``
    base_metric = 2347.78722  # 基准评估值

    # 获取训练过程的日志文件并计算最后一轮迭代的评估值
    log_file = os.path.join(output_dir, "train.log")
    last_metric = get_last_eval_metric(log_file, loss_function)

    assert float(last_metric) == base_metric


if __name__ == "__main__":
    # 使用 pytest 模块运行测试函数
    code = pytest.main([sys.argv[0]])
    sys.exit(code)
