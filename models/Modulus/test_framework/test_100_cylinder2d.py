import os
import re
import json
import sys
import time
import subprocess
import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tools.tools import run_model, plot_kpi_curves, log_parse


@pytest.mark.parametrize("model_name", ["examples-cylinder-cylinder_2d"])
def test_dy2st(model_name):
    # 执行torch
    pytorch_exit = run_model(model_name, 'pytorch')
    assert pytorch_exit == 0
    # 执行dy2st
    dy2st_exit = run_model(model_name, 'dy2st')
    assert dy2st_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    assert pytorch_kpi[0] != -1
    kpi_loss['pytorch'] = pytorch_kpi
    # 解析dy2st日志文件，提取损失值
    dy2st_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st.log', 'Loss')
    assert dy2st_kpi[0] != -1
    kpi_loss['dy2st'] = dy2st_kpi
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st')
    # 计算dy2st和pytorch的损失值差异
    diff_dy2st_1 = (dy2st_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    diff_dy2st_all = sum((dy2st_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    print("diff_dy2st_1:", diff_dy2st_1)
    print("diff_dy2st_all:", diff_dy2st_all)
    assert diff_dy2st_1 < 1e-6
    assert diff_dy2st_all < 1e-2


if __name__ == "__main__":
    current_date = datetime.now()
    code = pytest.main([f"--html=report_{current_date}.html", sys.argv[0]])
    sys.exit(code)
