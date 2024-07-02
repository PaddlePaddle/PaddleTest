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


@pytest.mark.parametrize("model_name", ["examples-ldc-ldc_2d",
                                      "examples-turbulent_channel-2d_std_wf-u_tau_lookup",
                                      "examples-turbulent_channel-2d_std_wf-u_tau_lookup"])
def test_all_step(model_name):
    # 执行torch
    pytorch_exit = run_model(model_name, 'pytorch', '')
    assert pytorch_exit == 0
    # 执行dy2st+prim
    dy2st_prim_exit = run_model(model_name, 'dy2st_prim','')
    assert dy2st_prim_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'loss:')
    assert pytorch_kpi[0] != -1
    kpi_loss['pytorch'] = pytorch_kpi
    # 解析dy2st+prim日志文件，提取损失值
    dy2st_prim_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st_prim.log', 'loss:')
    assert dy2st_prim_kpi[0] != -1
    kpi_loss['dy2st_prim'] = dy2st_prim_kpi
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st_prim')
    # 计算动转静+prim和pytorch的损失值差异
    diff_prim_1 = (dy2st_prim_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    diff_prim_all = sum((dy2st_prim_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    print("diff_prim_1:", diff_prim_1)
    print("diff_prim_all:", diff_prim_all)
    # 判断差异是否小于阈值
    assert diff_prim_1 < 1e-6
    assert diff_prim_all < 1e-5


if __name__ == "__main__":
    current_date = datetime.now()
    code = pytest.main([f"--html=report_{current_date}.html", sys.argv[0]])
    sys.exit(code)
