# 导入模块
import os
import sys
import subprocess

import pytest

from tools.log_analysis import get_last_epoch_loss


def test_darcy2d():
    """
    测试函数：测试 ldc2d_unsteady_Re10.py 脚本
    """
    # 定义变量
    output_dir = "./output_darcy2d"  # 输出目录
    epoch_num = 100  # 迭代次数
    base_loss = 8993487.00000  # 基准损失值
    py_version = os.getenv("py_version", "3.8")  # Python 版本号，从环境变量中获取，默认值为3.9

    # 如果输出目录已经存在，则删除
    if os.path.exists(output_dir):
        subprocess.run(f"rm -rf {output_dir}", shell=True, check=True)

    # 执行命令行命令，运行 ldc2d_unsteady_Re10.py 脚本
    command = (
        f"python{py_version} ../../examples/darcy/darcy2d.py --epochs={epoch_num} --output_dir={output_dir}"
    )
    subprocess.run(command, shell=True, check=True)

    # 获取训练过程的日志文件并计算最后一轮迭代的损失值
    log_file = os.path.join(output_dir, "train.log")
    last_loss = get_last_epoch_loss(log_file, epoch_num)

    # 断言最后一轮迭代的损失值是否等于基准损失值
    assert float(last_loss) == base_loss

if __name__ == "__main__":
    # 使用 pytest 模块运行测试函数
    code = pytest.main([sys.argv[0]])
    sys.exit(code)
