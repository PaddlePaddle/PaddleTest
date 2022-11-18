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
import time

import base


FONT = {
    "g": "00ff00",
    "b": "ff0000",
    "s": "000000",
    "o": "cccccc",
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
                "gsb": "o",
                "unit": info["jingdu"]["unit"],
            },
            "xingneng": {
                "th": info["xingneng"]["th"],
                "base": info["xingneng"]["value"],
                "benchmark": -1,
                "diff": -1,
                "gsb": "o",
                "unit": info["xingneng"]["unit"],
            },
        }

        if model not in benchmark_keys:
            continue

        compare_res[model]["jingdu"]["benchmark"] = benchmark_res[model]["jingdu"]["value"]
        gap = compare_res[model]["jingdu"]["benchmark"] - compare_res[model]["jingdu"]["base"]
        diff = gap / compare_res[model]["jingdu"]["base"]
        compare_res[model]["jingdu"]["diff"] = diff
        if diff == 0:
            compare_res[model]["jingdu"]["gsb"] = "s"
        elif diff < 0:
            compare_res[model]["jingdu"]["gsb"] = "b"
        elif diff > 0:
            compare_res[model]["jingdu"]["gsb"] = "g"

        compare_res[model]["xingneng"]["benchmark"] = benchmark_res[model]["xingneng"]["value"]
        gap = compare_res[model]["xingneng"]["benchmark"] - compare_res[model]["xingneng"]["base"]
        diff = gap / compare_res[model]["xingneng"]["base"]
        compare_res[model]["xingneng"]["diff"] = diff
        if diff == 0:
            compare_res[model]["xingneng"]["gsb"] = "s"
        elif diff < 0:
            compare_res[model]["xingneng"]["gsb"] = "b"
        elif diff > 0:
            compare_res[model]["xingneng"]["gsb"] = "g"

    return compare_res


def gsb(compare_res):
    """
    统计compare_res的gsb
    """
    gsb = {
        "jingdu": {
            "gsb": "",
            "g": 0,
            "s": 0,
            "b": 0,
            "total": 0,
            "b_ratio": 0,
            "g_ratio": 0,
        },
        "xingneng": {
            "gsb": "",
            "g": 0,
            "s": 0,
            "b": 0,
            "total": 0,
            "b_ratio": 0,
            "g_ratio": 0,
        },
    }
    for model, info in compare_res.items():
        if info["jingdu"]["gsb"] == "g":
            gsb["jingdu"]["g"] += 1
        elif info["jingdu"]["gsb"] == "s":
            gsb["jingdu"]["s"] += 1
        elif info["jingdu"]["gsb"] == "b":
            gsb["jingdu"]["b"] += 1
        gsb["jingdu"]["gsb"] = "{}:{}:{}".format(gsb["jingdu"]["g"], gsb["jingdu"]["s"], gsb["jingdu"]["b"])
        gsb["jingdu"]["total"] = gsb["jingdu"]["g"] + gsb["jingdu"]["s"] + gsb["jingdu"]["b"]
        if gsb["jingdu"]["total"] > 0:
            gsb["jingdu"]["b_ratio"] = gsb["jingdu"]["b"] / gsb["jingdu"]["total"]
            gsb["jingdu"]["g_ratio"] = gsb["jingdu"]["g"] / gsb["jingdu"]["total"]

        if info["xingneng"]["gsb"] == "g":
            gsb["xingneng"]["g"] += 1
        elif info["xingneng"]["gsb"] == "s":
            gsb["xingneng"]["s"] += 1
        elif info["xingneng"]["gsb"] == "b":
            gsb["xingneng"]["b"] += 1
        gsb["xingneng"]["gsb"] = "{}:{}:{}".format(gsb["xingneng"]["g"], gsb["xingneng"]["s"], gsb["xingneng"]["b"])
        gsb["xingneng"]["total"] = gsb["xingneng"]["g"] + gsb["xingneng"]["s"] + gsb["xingneng"]["b"]
        if gsb["xingneng"]["total"] > 0:
            gsb["xingneng"]["b_ratio"] = gsb["xingneng"]["b"] / gsb["xingneng"]["total"]
            gsb["xingneng"]["g_ratio"] = gsb["xingneng"]["g"] / gsb["xingneng"]["total"]
    return gsb


def res_summary(trt_int8, trt_fp16, mkldnn_int8, mkldnn_fp32):
    """
    汇总不同模式下的数据
    """
    tongji = {}
    res = {}

    # 统计数据
    gsb_trt_int8 = gsb(trt_int8)
    gsb_trt_fp16 = gsb(trt_fp16)
    gsb_mkldnn_int8 = gsb(mkldnn_int8)
    gsb_mkldnn_fp32 = gsb(mkldnn_fp32)
    gsb_total = {
        "jingdu": {
            "gsb": "",
            "g": 0,
            "s": 0,
            "b": 0,
            "total": 0,
            "b_ratio": 0,
            "g_ratio": 0,
        },
        "xingneng": {
            "gsb": "",
            "g": 0,
            "s": 0,
            "b": 0,
            "total": 0,
            "b_ratio": 0,
            "g_ratio": 0,
        },
    }
    gsb_total["jingdu"]["g"] = (
        gsb_trt_int8["jingdu"]["g"]
        + gsb_trt_fp16["jingdu"]["g"]
        + gsb_mkldnn_int8["jingdu"]["g"]
        + gsb_mkldnn_fp32["jingdu"]["g"]
    )
    gsb_total["jingdu"]["s"] = (
        gsb_trt_int8["jingdu"]["s"]
        + gsb_trt_fp16["jingdu"]["s"]
        + gsb_mkldnn_int8["jingdu"]["s"]
        + gsb_mkldnn_fp32["jingdu"]["s"]
    )
    gsb_total["jingdu"]["b"] = (
        gsb_trt_int8["jingdu"]["b"]
        + gsb_trt_fp16["jingdu"]["b"]
        + gsb_mkldnn_int8["jingdu"]["b"]
        + gsb_mkldnn_fp32["jingdu"]["b"]
    )
    gsb_total["jingdu"]["total"] = (
        gsb_trt_int8["jingdu"]["total"]
        + gsb_trt_fp16["jingdu"]["total"]
        + gsb_mkldnn_int8["jingdu"]["total"]
        + gsb_mkldnn_fp32["jingdu"]["total"]
    )
    gsb_total["jingdu"]["gsb"] = "{}:{}:{}".format(
        gsb_total["jingdu"]["g"], gsb_total["jingdu"]["s"], gsb_total["jingdu"]["b"]
    )
    if gsb_total["jingdu"]["total"] > 0:
        gsb_total["jingdu"]["b_ratio"] = gsb_total["jingdu"]["b"] / gsb_total["jingdu"]["total"]
        gsb_total["jingdu"]["g_ratio"] = gsb_total["jingdu"]["g"] / gsb_total["jingdu"]["total"]
    gsb_total["xingneng"]["g"] = (
        gsb_trt_int8["xingneng"]["g"]
        + gsb_trt_fp16["xingneng"]["g"]
        + gsb_mkldnn_int8["xingneng"]["g"]
        + gsb_mkldnn_fp32["xingneng"]["g"]
    )
    gsb_total["xingneng"]["s"] = (
        gsb_trt_int8["xingneng"]["s"]
        + gsb_trt_fp16["xingneng"]["s"]
        + gsb_mkldnn_int8["xingneng"]["s"]
        + gsb_mkldnn_fp32["xingneng"]["s"]
    )
    gsb_total["xingneng"]["b"] = (
        gsb_trt_int8["xingneng"]["b"]
        + gsb_trt_fp16["xingneng"]["b"]
        + gsb_mkldnn_int8["xingneng"]["b"]
        + gsb_mkldnn_fp32["xingneng"]["b"]
    )
    gsb_total["xingneng"]["total"] = (
        gsb_trt_int8["xingneng"]["total"]
        + gsb_trt_fp16["xingneng"]["total"]
        + gsb_mkldnn_int8["xingneng"]["total"]
        + gsb_mkldnn_fp32["xingneng"]["total"]
    )
    gsb_total["xingneng"]["gsb"] = "{}:{}:{}".format(
        gsb_total["xingneng"]["g"], gsb_total["xingneng"]["s"], gsb_total["xingneng"]["b"]
    )
    if gsb_total["xingneng"]["total"] > 0:
        gsb_total["xingneng"]["b_ratio"] = gsb_total["xingneng"]["b"] / gsb_total["xingneng"]["total"]
        gsb_total["xingneng"]["g_ratio"] = gsb_total["xingneng"]["g"] / gsb_total["xingneng"]["total"]

    tongji = {
        "trt_int8": gsb_trt_int8,
        "trt_fp16": gsb_trt_fp16,
        "mkldnn_int8": gsb_mkldnn_int8,
        "mkldnn_fp32": gsb_mkldnn_fp32,
        "total": gsb_total,
    }
    
    # 详细数据
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

    return res, tongji


def res2xls(env, res, tongji):
    """
    将结果保存为excel文件
    """
    wb = openpyxl.Workbook()

    # table1: 详细数据
    sheet_detail = wb.create_sheet(index=0, title="detail")
    sheet_detail = wb["detail"]

    # 表头
    sheet_detail.cell(1, 2).value = env

    sheet_detail.cell(2, 2).value = "trt_int8"
    sheet_detail.cell(2, 10).value = "trt_fp16"
    sheet_detail.cell(2, 18).value = "mkldnn_int8"
    sheet_detail.cell(2, 26).value = "mkldnn_fp32"

    sheet_detail.cell(3, 2).value = "精度"
    sheet_detail.cell(3, 6).value = "性能"
    sheet_detail.cell(3, 10).value = "精度"
    sheet_detail.cell(3, 14).value = "性能"
    sheet_detail.cell(3, 18).value = "精度"
    sheet_detail.cell(3, 22).value = "性能"
    sheet_detail.cell(3, 26).value = "精度"
    sheet_detail.cell(3, 30).value = "性能"

    column_s = 2
    for i in range(8):
        sheet_detail.cell(4, column_s).value = "base值"
        sheet_detail.cell(4, column_s + 1).value = "实际值"
        sheet_detail.cell(4, column_s + 2).value = "阈值"
        sheet_detail.cell(4, column_s + 3).value = "diff"
        column_s += 4

    # 数据
    row_s = 5
    for model, info in res.items():
        column_s = 1
        sheet_detail.cell(row_s, column_s).value = model
        # trt_int8
        sheet_detail.cell(row_s, column_s + 1).value = info["trt_int8"]["jingdu"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["trt_int8"]["jingdu"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["trt_int8"]["jingdu"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["trt_int8"]["jingdu"]["diff"]
        _color = info["trt_int8"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet_detail.cell(row_s, column_s + 1).value = info["trt_int8"]["xingneng"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["trt_int8"]["xingneng"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["trt_int8"]["xingneng"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["trt_int8"]["xingneng"]["diff"]
        _color = info["trt_int8"]["xingneng"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4
        # trt_fp16
        sheet_detail.cell(row_s, column_s + 1).value = info["trt_fp16"]["jingdu"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["trt_fp16"]["jingdu"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["trt_fp16"]["jingdu"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["trt_fp16"]["jingdu"]["diff"]
        _color = info["trt_fp16"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet_detail.cell(row_s, column_s + 1).value = info["trt_fp16"]["xingneng"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["trt_fp16"]["xingneng"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["trt_fp16"]["xingneng"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["trt_fp16"]["xingneng"]["diff"]
        _color = info["trt_fp16"]["xingneng"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4
        # mkldnn_int8
        sheet_detail.cell(row_s, column_s + 1).value = info["mkldnn_int8"]["jingdu"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["mkldnn_int8"]["jingdu"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["mkldnn_int8"]["jingdu"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["mkldnn_int8"]["jingdu"]["diff"]
        _color = info["mkldnn_int8"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet_detail.cell(row_s, column_s + 1).value = info["mkldnn_int8"]["xingneng"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["mkldnn_int8"]["xingneng"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["mkldnn_int8"]["xingneng"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["mkldnn_int8"]["xingneng"]["diff"]
        _color = info["mkldnn_int8"]["xingneng"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4
        # mkldnn_fp32
        sheet_detail.cell(row_s, column_s + 1).value = info["mkldnn_fp32"]["jingdu"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["mkldnn_fp32"]["jingdu"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["mkldnn_fp32"]["jingdu"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["mkldnn_fp32"]["jingdu"]["diff"]
        _color = info["mkldnn_fp32"]["jingdu"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4
        sheet_detail.cell(row_s, column_s + 1).value = info["mkldnn_fp32"]["xingneng"]["base"]
        sheet_detail.cell(row_s, column_s + 2).value = info["mkldnn_fp32"]["xingneng"]["benchmark"]
        sheet_detail.cell(row_s, column_s + 3).value = info["mkldnn_fp32"]["xingneng"]["th"]
        sheet_detail.cell(row_s, column_s + 4).value = info["mkldnn_fp32"]["xingneng"]["diff"]
        _color = info["mkldnn_fp32"]["xingneng"]["gsb"]
        _font = Font(color=FONT[_color])
        sheet_detail.cell(row_s, column_s + 4).font = _font
        column_s += 4

        row_s += 1

    # 并表头单元格
    sheet_detail.merge_cells("B1:Z1")
    sheet_detail.merge_cells("B2:I2")
    sheet_detail.merge_cells("B3:E3")
    sheet_detail.merge_cells("F3:I3")
    sheet_detail.merge_cells("J2:Q2")
    sheet_detail.merge_cells("J3:M3")
    sheet_detail.merge_cells("N3:Q3")
    sheet_detail.merge_cells("R2:Y2")
    sheet_detail.merge_cells("R3:U3")
    sheet_detail.merge_cells("V3:Y3")
    sheet_detail.merge_cells("Z2:AG2")
    sheet_detail.merge_cells("Z3:AC3")
    sheet_detail.merge_cells("AD3:AG3")

    # table2: gsb统计数据
    sheet_gsb = wb.create_sheet(index=1, title="gsb")
    sheet_gsb = wb["gsb"]

    # 表头
    sheet_gsb.cell(1, 2).value = "精度"
    sheet_gsb.cell(1, 5).value = "性能"
    
    sheet_gsb.cell(2, 2).value = "GSB"
    sheet_gsb.cell(2, 3).value = "下降数（占比）"
    sheet_gsb.cell(2, 4).value = "上升数（占比）"
    sheet_gsb.cell(2, 5).value = "GSB"
    sheet_gsb.cell(2, 6).value = "下降数（占比）"
    sheet_gsb.cell(2, 7).value = "上升数（占比）"

    sheet_gsb.cell(3, 1).value = "trt_int8"
    sheet_gsb.cell(4, 1).value = "trt_fp16"
    sheet_gsb.cell(5, 1).value = "mkldnn_int8"
    sheet_gsb.cell(6, 1).value = "mkldnn_fp32"
    sheet_gsb.cell(7, 1).value = "total"

    # 数据
    column_s = 1
    row_s = 3
    for mode, info in tongji.items():
        sheet_gsb.cell(row_s, column_s).value = mode
        sheet_gsb.cell(row_s, column_s + 1).value = info["jingdu"]["gsb"]
        sheet_gsb.cell(row_s, column_s + 2).value = "{}({})".format(info["jingdu"]["b"], info["jingdu"]["b_ratio"])
        sheet_gsb.cell(row_s, column_s + 3).value = "{}({})".format(info["jingdu"]["g"], info["jingdu"]["g_ratio"])
        sheet_gsb.cell(row_s, column_s + 4).value = info["xingneng"]["gsb"]
        sheet_gsb.cell(row_s, column_s + 5).value = "{}({})".format(info["xingneng"]["b"], info["xingneng"]["b_ratio"])
        sheet_gsb.cell(row_s, column_s + 6).value = "{}({})".format(info["xingneng"]["g"], info["xingneng"]["g_ratio"])
        column_s = 1
        row_s += 1

    # 并表头单元格
    sheet_gsb.merge_cells("B1:D1")
    sheet_gsb.merge_cells("E1:G1")

    save_file = sys.argv[5]
    wb.save("{}".format(save_file))


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

    res, tongji = res_summary(trt_int8, trt_fp16, mkldnn_int8, mkldnn_int8)

    env = "环境\n"
    env += "docker: "
    env += sys.argv[1]
    env += "\n"
    env += "paddle_branch: "
    env += sys.argv[2]
    env += "\n"
    env += "paddle_commit: "
    env += sys.argv[3]
    env += "\n"
    env += "device: "
    env += sys.argv[4]
    env += "\n"

    res2xls(env, res, tongji)


if __name__ == "__main__":
    run()
