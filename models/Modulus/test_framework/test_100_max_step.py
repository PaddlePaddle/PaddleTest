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

model_list = []
with open('./model.json', 'r', encoding='utf-8') as f:
    # 读取JSON数据
    test_json = json.load(f)
    # 遍历 JSON 对象获取所有的键
    for key in test_json.keys():
        model_list.append(key)

@pytest.mark.parametrize("model_name", model_list)
def test_dynamic(model_name):
    # 执行torch
    pytorch_exit = run_model(model_name, 'pytorch')
    assert pytorch_exit == 0
    # 执行动态图
    dynamic_exit = run_model(model_name, 'dynamic')
    assert dynamic_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    assert pytorch_kpi[0] != -1
    kpi_loss['pytorch'] = pytorch_kpi
    # 解析动态图日志文件，提取损失值
    dynamic_kpi = log_parse(f'./logs/{model_name}/{model_name}_dynamic.log', 'Loss')
    assert dynamic_kpi[0] != -1
    kpi_loss['dynamic'] = dynamic_kpi
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dynamic')
    # 计算动态图和pytorch的损失值差异
    diff_dynamic_1 = (dynamic_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    diff_dynamic_all = sum((dynamic_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    print("diff_dynamic_1:", diff_dynamic_1)
    print("diff_dynamic_all:", diff_dynamic_all)
    # 判断差异是否小于阈值
    assert diff_dynamic_1 < 1e-6
    assert diff_dynamic_all < 1e-5

@pytest.mark.parametrize("model_name", model_list)
def test_dy2st(model_name):
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
    assert diff_dy2st_all < 1e-5

@pytest.mark.parametrize("model_name", model_list)
def test_dy2st_prim(model_name):
    # 执行dy2st+prim
    dy2st_prim_exit = run_model(model_name, 'dy2st_prim')
    assert dy2st_prim_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    assert pytorch_kpi[0] != -1
    kpi_loss['pytorch'] = pytorch_kpi
    dy2st_prim_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st_prim.log', 'Loss')
    assert dy2st_prim_kpi[0] != -1
    kpi_loss['dy2st_prim'] = dy2st_prim_kpi
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st_prim')
    # 计算动转静+prim和pytorch的损失值差异
    diff_prim_1 = (dy2st_prim_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    diff_prim_all = sum((dy2st_prim_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    print("diff_prim_1:", diff_prim_1)
    print("diff_prim_all:", diff_prim_all)
    assert diff_prim_1 < 1e-6
    assert diff_prim_all < 1e-5

if __name__ == "__main__":
    # 定义日志文件路径
    # log_content = "../test_bracket_dynamic.log"
    # # 定义KPI名称
    # kpi_name = "Loss"
    # # 调用log_parse函数，提取KPI值
    # kpi_value = log_parse(log_content, kpi_name)
    # print("KPI Value:", str(kpi_value))
    current_date = datetime.now()
    code = pytest.main([f"--html=report_{current_date}.html", sys.argv[0]])
    sys.exit(code)
