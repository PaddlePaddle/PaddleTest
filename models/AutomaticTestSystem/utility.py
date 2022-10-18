# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2022/9/2 3:46 PM
  * @brief  model metric
  *
  **************************************************************************/
"""
import subprocess
import re
import ast
import logging
import os
import os.path
import platform
import filecmp
import allure
from plot_paddle_torch import *
import chardet
import paddle
import numpy as np
import yaml
import pytest
from pytest_assume.plugin import assume
from pytest import approx


def exit_check_fucntion(exit_code, output, mode, log_dir=""):
    """
    exit_check_fucntion
    """
    print(output)
    if exit_code == 0:
        allure.attach(output, "output.log", allure.attachment_type.TEXT)
    assert exit_code == 0, " %s  model pretrained failed!   log information:%s" % (mode, output)
    assert "Error" not in output, "%s  model failed!   log information:%s" % (mode, output)
    if "ABORT!!!" in output:
        log_dir = os.path.abspath(log_dir)
        all_files = os.listdir(log_dir)
        for file in all_files:
            print(file)
            filename = os.path.join(log_dir, file)
            with open(filename) as file_obj:
                content = file_obj.read()
                print(content)
    assert "ABORT!!!" not in output, "%s  model failed!   log information:%s" % (mode, output)
    logging.info("train model sucessfuly!")


def check_charset(file_path):
    """
    check_charset
    """
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)["encoding"]
    return charset


def allure_attach(filename, name, fileformat):
    """
    allure_attach
    """
    with open(filename, mode="rb") as f:
        file_content = f.read()
    allure.attach(file_content, name=name, attachment_type=fileformat)


def allure_step(cmd, output):
    """
    allure_step
    """
    with allure.step("运行指令：{}".format(cmd)):
        pass


def readfile(filename):
    """
    readfile
    """
    with open(filename, mode="r", encoding="utf-8") as f:
        text = f.readline()
    return text


def check_infer_metric(category, output, dataset):
    """
    check_infer_metric
    """
    if category == "rec":
        metric = metricExtraction("result", output)
        rec_docs = metric.strip().split("\t")[0]
        rec_scores = metric.strip().split("\t")[1]
        rec_scores = float(rec_scores)

        print("rec_docs:{}".format(rec_docs))
        print("rec_scores:{}".format(rec_scores))

        expect_rec_docs = "joint"
        expect_rec_scores = 0.9999

        with assume:
            assert rec_docs == expect_rec_docs, (
                "check rec_docs failed! real rec_docs is: %s,\
                            expect rec_docs is: %s"
                % (rec_docs, expect_rec_docs)
            )
        with assume:
            assert rec_scores == approx(expect_rec_scores, abs=1e-2), (
                "check rec_scores failed!   real rec_scores is: %s, \
                            expect rec_scores is: %s"
                % (rec_scores, expect_rec_scores)
            )
        print("*************************************************************************")

    elif category == "det":
        allure_attach(
            "PaddleOCR/checkpoints/det_db/det_results/img_10.jpg",
            "checkpoints/det_db/det_results/img_10.jpg",
            allure.attachment_type.JPG,
        )
        allure_attach(
            "PaddleOCR/checkpoints/det_db/predicts_db.txt",
            "checkpoints/det_db/predicts_db.txt",
            allure.attachment_type.TEXT,
        )
        allure_attach(
            "./metric/predicts_db_" + dataset + ".txt",
            "./metric/predicts_db_" + dataset + ".txt",
            allure.attachment_type.TEXT,
        )
        real_det_bbox = readfile("PaddleOCR/checkpoints/det_db/predicts_db.txt")
        expect_det_bbox = readfile("./metric/predicts_db_" + dataset + ".txt")
        assert real_det_bbox == expect_det_bbox, "real det_bbox should equal expect det_bbox"
    elif category == "table":
        real_metric = metricExtraction("result", output)
        table_bbox = real_metric.split("'</html>'],")[0]
        print("table_bbox:{}".format(table_bbox))
        allure.attach(real_metric, "real_table_result", allure.attachment_type.TEXT)
        allure_attach("./metric/infer_table.txt", "./metric/infer_table.txt", allure.attachment_type.TEXT)

        real_table = real_metric
        expect_table = readfile("./metric/infer_table.txt")
        print(len(real_table))
        print("expect_table:{}".format(expect_table))
        print(len(expect_table))
    elif category == "sr":
        allure_attach(
            "PaddleOCR/infer_result/sr_word_52.png", "infer_result/sr_word_52.png", allure.attachment_type.PNG
        )
    elif category == "kie/vi_layoutxlm":
        if os.path.exists("./output/ser/xfund_zh/res/zh_val_0_ser.jpg"):
            allure_attach(
                "PaddleOCR/output/ser/xfund_zh/res/zh_val_0_ser.jpg",
                "./output/output/ser/xfund_zh/res/zh_val_0_ser.jpg",
                allure.attachment_type.JPG,
            )
        if os.path.exists("./output/re/xfund_zh/with_gt/zh_val_0_ser_re.jpg"):
            allure_attach(
                "PaddleOCR/output/re/xfund_zh/with_gt/zh_val_0_ser_re.jpg",
                "./output/re/xfund_zh/with_gt/zh_val_0_ser_re.jpg",
                allure.attachment_type.JPG,
            )
    else:
        pass


def check_predict_metric(category, output, dataset):
    """
    check_predict_metric
    """
    if category == "rec":
        for line in output.split("\n"):
            if "Predicts of" in line:
                output_rec = line
        output_rec_list = re.findall(r"\((.*?)\)", output_rec)
        print(output_rec_list)
        rec_docs = output_rec_list[0].split(",")[0].strip("'")
        rec_scores = output_rec_list[0].split(",")[1]
        rec_scores = float(rec_scores)

        print("rec_docs:{}".format(rec_docs))
        print("rec_scores:{}".format(rec_scores))
        expect_rec_docs = "super"
        expect_rec_scores = 0.9999
        with assume:
            assert rec_docs == expect_rec_docs, (
                "check rec_docs failed! real rec_docs is: %s,\
                            expect rec_docs is: %s"
                % (rec_docs, expect_rec_docs)
            )
        with assume:
            assert rec_scores == approx(expect_rec_scores, abs=1e-2), (
                "check rec_scores failed!   real rec_scores is: %s, \
                            expect rec_scores is: %s"
                % (rec_scores, expect_rec_scores)
            )
        print("*************************************************************************")
    elif category == "det":
        allure_attach(
            "PaddleOCR/inference_results/det_res_img_10.jpg",
            "inference_results/det_res_img_10.jpg",
            allure.attachment_type.JPG,
        )
        allure_attach(
            "PaddleOCR/inference_results/det_results.txt",
            "inference_results/det_results.txt",
            allure.attachment_type.TEXT,
        )
        for line in output.split("\n"):
            if "img_10.jpg" in line:
                output_det = line
                print(output_det)
                break

        det_bbox = output_det.split("\t")[-1]
        det_bbox = ast.literal_eval(det_bbox)
        print("det_bbox:{}".format(det_bbox))
        if dataset == "icdar15":
            expect_det_bbox = [
                [[39.0, 88.0], [147.0, 80.0], [149.0, 103.0], [41.0, 110.0]],
                [[149.0, 82.0], [199.0, 79.0], [200.0, 98.0], [150.0, 101.0]],
                [[35.0, 54.0], [97.0, 54.0], [97.0, 78.0], [35.0, 78.0]],
                [[100.0, 53.0], [141.0, 53.0], [141.0, 79.0], [100.0, 79.0]],
                [[181.0, 54.0], [204.0, 54.0], [204.0, 73.0], [181.0, 73.0]],
                [[139.0, 54.0], [187.0, 50.0], [189.0, 75.0], [141.0, 79.0]],
                [[193.0, 29.0], [253.0, 29.0], [253.0, 48.0], [193.0, 48.0]],
                [[161.0, 28.0], [200.0, 28.0], [200.0, 48.0], [161.0, 48.0]],
                [[107.0, 21.0], [161.0, 24.0], [159.0, 49.0], [105.0, 46.0]],
                [[29.0, 19.0], [107.0, 19.0], [107.0, 46.0], [29.0, 46.0]],
            ]
        else:
            expect_det_bbox = [
                [[42.0, 89.0], [201.0, 79.0], [202.0, 98.0], [43.0, 108.0]],
                [[32.0, 56.0], [206.0, 53.0], [207.0, 75.0], [32.0, 78.0]],
                [[18.0, 22.0], [251.0, 31.0], [250.0, 49.0], [17.0, 41.0]],
            ]

        with assume:
            assert np.array(det_bbox) == approx(np.array(expect_det_bbox), abs=2), (
                "check det_bbox failed!  \
                           real det_bbox is: %s, expect det_bbox is: %s"
                % (det_bbox, expect_det_bbox)
            )
        print("*************************************************************************")
    elif category == "table":
        allure_attach("PaddleOCR/output/table.jpg", "./output/table.jpg", allure.attachment_type.JPG)
    elif category == "sr":
        allure_attach(
            "PaddleOCR/infer_result/sr_word_52.png", "infer_result/sr_word_52.png", allure.attachment_type.PNG
        )
    else:
        pass
