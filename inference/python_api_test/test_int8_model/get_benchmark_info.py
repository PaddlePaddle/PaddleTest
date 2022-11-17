"""
get benchmark info from log
"""

import os
import sys
import json
import openpyxl
from openpyxl.styles import Font
from openpyxl.styles import PatternFill
from openpyxl.styles import Alignment

import base


FONT = {
    "g": "00ff00",
    "b": "ff0000",
    "s": "000000",
}


def get_runtime_info(log_file):
    """
    获取本次执行结果
    """
    benchmark_res = {}
    if not os.path.exists(log_file):
        return benchmark_res
    with open(log_file) as fin:
        benchmark_lines = ""
        lines = fin.readlines()
        for line in lines:
            if "[Benchmark][final result]" in line:
                tmp = line.strip("[Benchmark][final result]").strip()
                res_json = eval(tmp)
                model_name = res_json["model_name"]
                benchmark_res[model_name] = res_json
    return benchmark_res


def get_base_info(mode):
    """
    从base文件中读取base数据
    mode: trt_int8 trt_fp16 mkldnn_int8 mkldnn_fp32
    """
    print(mode)
    if mode == "trt_int8":
        base_res = base.trt_int8
    elif mode == "trt_fp16":
        base_res = base.trt_fp16
    elif mode == "mkldnn_int8":
        base_res = base.mkldnn_int8
    elif mode == "mkldnn_fp32":
        base_res = base.mkldnn_fp32
    else:
        base_res = None
    return base_res


def compare_diff(base_res, benchmark_res):
    """
    计算本次结果与base的diff、gsb
    """
    benchmark_keys = benchmark_res.keys()
    compare_res = {}
    for model, info in base_res.items():
        compare_res[model] = {
            "jingdu": {
                "th": info["jingdu"]["th"],
                "base": info["jingdu"]["value"],
                "benchmark": -1,
                "diff": -1,
                "gsb": "s",
                "unit": info["jingdu"]["unit"],
            },
            "xingneng": {
                "th": info["xingneng"]["th"],
                "base": info["xingneng"]["value"],
                "benchmark": -1,
                "diff": -1,
                "gsb": "s",
                "unit": info["xingneng"]["unit"],
            },
        }

        if model not in benchmark_keys:
            continue

        compare_res[model]["jingdu"]["benchmark"] = benchmark_res[model]["jingdu"]["value"]
        gap = compare_res[model]["jingdu"]["benchmark"] - compare_res[model]["jingdu"]["base"]
        diff = gap / compare_res[model]["jingdu"]["base"]
        compare_res[model]["jingdu"]["diff"] = diff
        if gap < 0:
            compare_res[model]["jingdu"]["gsb"] = "b"
        elif gap > 0:
            compare_res[model]["jingdu"]["gsb"] = "g"

        compare_res[model]["xingneng"]["benchmark"] = benchmark_res[model]["xingneng"]["value"]
        gap = compare_res[model]["xingneng"]["benchmark"] - compare_res[model]["xingneng"]["base"]
        diff = gap / compare_res[model]["xingneng"]["base"]
        compare_res[model]["xingneng"]["diff"] = diff
        if gap < 0:
            compare_res[model]["xingneng"]["gsb"] = "b"
        elif gap > 0:
            compare_res[model]["xingneng"]["gsb"] = "g"

    return compare_res


def res_summary(trt_int8, trt_fp16, mkldnn_int8, mkldnn_fp32):
    """
    汇总不同模式下的数据
    """
    res = {}
    models = trt_int8.keys()
    for model in models:
        res[model] = {
            "trt_int8": {
                "jingdu": {
                    "th": trt_int8[model]["jingdu"]["th"],
                    "base": trt_int8[model]["jingdu"]["base"],
                    "benchmark": trt_int8[model]["jingdu"]["benchmark"],
                    "diff": trt_int8[model]["jingdu"]["diff"],
                    "gsb": trt_int8[model]["jingdu"]["gsb"],
                    "unit": trt_int8[model]["jingdu"]["unit"],
                },
                "xingneng": {
                    "th": trt_int8[model]["xingneng"]["th"],
                    "base": trt_int8[model]["xingneng"]["base"],
                    "benchmark": trt_int8[model]["xingneng"]["benchmark"],
                    "diff": trt_int8[model]["xingneng"]["diff"],
                    "gsb": trt_int8[model]["xingneng"]["gsb"],
                    "unit": trt_int8[model]["xingneng"]["unit"],
                },
            },
            "trt_fp16": {
                "jingdu": {
                    "th": trt_fp16[model]["jingdu"]["th"],
                    "base": trt_fp16[model]["jingdu"]["base"],
                    "benchmark": trt_fp16[model]["jingdu"]["benchmark"],
                    "diff": trt_fp16[model]["jingdu"]["diff"],
                    "gsb": trt_fp16[model]["jingdu"]["gsb"],
                    "unit": trt_fp16[model]["jingdu"]["unit"],
                },
                "xingneng": {
                    "th": trt_fp16[model]["xingneng"]["th"],
                    "base": trt_fp16[model]["xingneng"]["base"],
                    "benchmark": trt_fp16[model]["xingneng"]["benchmark"],
                    "diff": trt_fp16[model]["xingneng"]["diff"],
                    "gsb": trt_fp16[model]["xingneng"]["gsb"],
                    "unit": trt_fp16[model]["xingneng"]["unit"],
                },
            },
            "mkldnn_int8": {
                "jingdu": {
                    "th": mkldnn_int8[model]["jingdu"]["th"],
                    "base": mkldnn_int8[model]["jingdu"]["base"],
                    "benchmark": mkldnn_int8[model]["jingdu"]["benchmark"],
                    "diff": mkldnn_int8[model]["jingdu"]["diff"],
                    "gsb": mkldnn_int8[model]["jingdu"]["gsb"],
                    "unit": mkldnn_int8[model]["jingdu"]["unit"],
                },
                "xingneng": {
                    "th": mkldnn_int8[model]["xingneng"]["th"],
                    "base": mkldnn_int8[model]["xingneng"]["base"],
                    "benchmark": mkldnn_int8[model]["xingneng"]["benchmark"],
                    "diff": mkldnn_int8[model]["xingneng"]["diff"],
                    "gsb": mkldnn_int8[model]["xingneng"]["gsb"],
                    "unit": mkldnn_int8[model]["xingneng"]["unit"],
                },
            },
            "mkldnn_fp32": {
                "jingdu": {
                    "th": mkldnn_fp32[model]["jingdu"]["th"],
                    "base": mkldnn_fp32[model]["jingdu"]["base"],
                    "benchmark": mkldnn_fp32[model]["jingdu"]["benchmark"],
                    "diff": mkldnn_fp32[model]["jingdu"]["diff"],
                    "gsb": mkldnn_fp32[model]["jingdu"]["gsb"],
                    "unit": mkldnn_fp32[model]["jingdu"]["unit"],
                },
                "xingneng": {
                    "th": mkldnn_fp32[model]["xingneng"]["th"],
                    "base": mkldnn_fp32[model]["xingneng"]["base"],
                    "benchmark": mkldnn_fp32[model]["xingneng"]["benchmark"],
                    "diff": mkldnn_fp32[model]["xingneng"]["diff"],
                    "gsb": mkldnn_fp32[model]["xingneng"]["gsb"],
                    "unit": mkldnn_fp32[model]["xingneng"]["unit"],
                },
            },
        }

    return res


def res2xls(res):
    """
    将结果保存为excel文件
    """
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "工作表1"

    # 表头
    sheet.cell(1, 2).value = "环境"

    sheet.cell(2, 2).value = "trt_int8"
    sheet.cell(2, 10).value = "trt_fp16"
    sheet.cell(2, 18).value = "mkldnn_int8"
    sheet.cell(2, 26).value = "mkldnn_fp32"

    sheet.cell(3, 2).value = "精度"
    sheet.cell(3, 6).value = "性能"
    sheet.cell(3, 10).value = "精度"
    sheet.cell(3, 14).value = "性能"
    sheet.cell(3, 18).value = "精度"
    sheet.cell(3, 22).value = "性能"
    sheet.cell(3, 26).value = "精度"
    sheet.cell(3, 30).value = "性能"

    column_s = 2
    for i in range(8):
        sheet.cell(4, column_s).value = "base值"
        sheet.cell(4, column_s + 1).value = "实际值"
        sheet.cell(4, column_s + 2).value = "阈值"
        sheet.cell(4, column_s + 3).value = "diff"
        column_s += 4

    # 数据
    row_s = 5
    for model, info in res.items():
        column_s = 1
        sheet.cell(row_s, column_s).value = model
        # trt_int8
        sheet.cell(row_s, column_s + 1).value = info["trt_int8"]["jingdu"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["trt_int8"]["jingdu"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["trt_int8"]["jingdu"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["trt_int8"]["jingdu"]["diff"]
        _color = info["trt_int8"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet.cell(row_s, column_s + 1).value = info["trt_int8"]["xingneng"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["trt_int8"]["xingneng"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["trt_int8"]["xingneng"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["trt_int8"]["xingneng"]["diff"]
        _color = info["trt_int8"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4
        # trt_fp16
        sheet.cell(row_s, column_s + 1).value = info["trt_fp16"]["jingdu"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["trt_fp16"]["jingdu"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["trt_fp16"]["jingdu"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["trt_fp16"]["jingdu"]["diff"]
        _color = info["trt_fp16"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet.cell(row_s, column_s + 1).value = info["trt_fp16"]["xingneng"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["trt_fp16"]["xingneng"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["trt_fp16"]["xingneng"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["trt_fp16"]["xingneng"]["diff"]
        _color = info["trt_fp16"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4
        # mkldnn_int8
        sheet.cell(row_s, column_s + 1).value = info["mkldnn_int8"]["jingdu"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["mkldnn_int8"]["jingdu"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["mkldnn_int8"]["jingdu"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["mkldnn_int8"]["jingdu"]["diff"]
        _color = info["mkldnn_int8"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet.cell(row_s, column_s + 1).value = info["mkldnn_int8"]["xingneng"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["mkldnn_int8"]["xingneng"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["mkldnn_int8"]["xingneng"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["mkldnn_int8"]["xingneng"]["diff"]
        _color = info["mkldnn_int8"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4
        # mkldnn_fp32
        sheet.cell(row_s, column_s + 1).value = info["mkldnn_fp32"]["jingdu"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["mkldnn_fp32"]["jingdu"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["mkldnn_fp32"]["jingdu"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["mkldnn_fp32"]["jingdu"]["diff"]
        _color = info["mkldnn_fp32"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet.cell(row_s, column_s + 1).value = info["mkldnn_fp32"]["xingneng"]["base"]
        sheet.cell(row_s, column_s + 2).value = info["mkldnn_fp32"]["xingneng"]["benchmark"]
        sheet.cell(row_s, column_s + 3).value = info["mkldnn_fp32"]["xingneng"]["th"]
        sheet.cell(row_s, column_s + 4).value = info["mkldnn_fp32"]["xingneng"]["diff"]
        _color = info["mkldnn_fp32"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet.cell(row_s, column_s + 4).font = _font
        column_s += 4

        row_s += 1

    # 并表头单元格
    sheet.merge_cells("B1:Z1")
    sheet.merge_cells("B2:I2")
    sheet.merge_cells("B3:E3")
    sheet.merge_cells("F3:I3")
    sheet.merge_cells("J2:Q2")
    sheet.merge_cells("J3:M3")
    sheet.merge_cells("N3:Q3")
    sheet.merge_cells("R2:Y2")
    sheet.merge_cells("R3:U3")
    sheet.merge_cells("V3:Y3")
    sheet.merge_cells("Z2:AG2")
    sheet.merge_cells("Z3:AC3")
    sheet.merge_cells("AD3:AG3")

    wb.save("test.xlsx")


def run():
    """
    统计结果并保存到excel文件
    """
    # trt_int8
    log_file = "eval_trt_int8_acc.log"
    mode = "trt_int8"
    benchmark_res = get_runtime_info(log_file)
    base_res = get_base_info(mode)
    trt_int8 = compare_diff(base_res, benchmark_res)

    # trt_fp16
    log_file = "eval_trt_fp16_acc.log"
    mode = "trt_fp16"
    benchmark_res = get_runtime_info(log_file)
    base_res = get_base_info(mode)
    trt_fp16 = compare_diff(base_res, benchmark_res)

    # mkldnn_int8
    log_file = "eval_mkldnn_int8_acc.log"
    mode = "mkldnn_int8"
    benchmark_res = get_runtime_info(log_file)
    base_res = get_base_info(mode)
    mkldnn_int8 = compare_diff(base_res, benchmark_res)

    # mkldnn_fp32
    log_file = "eval_mkldnn_fp32_acc.log"
    mode = "mkldnn_fp32"
    benchmark_res = get_runtime_info(log_file)
    base_res = get_base_info(mode)
    mkldnn_fp32 = compare_diff(base_res, benchmark_res)

    res = res_summary(trt_int8, trt_fp16, mkldnn_int8, mkldnn_int8)

    res2xls(res)


if __name__ == "__main__":
    run()
