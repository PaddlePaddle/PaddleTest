# -*- coding: utf-8 -*-
import os
import ast
import re
import json
import argparse
import numpy as np

"python infer accuracy case"


def get_main_txt_data(log_file, model_type, model_name=None):
    """
    获取预测日志中检测框的 classid,confidence,boxcoordinates
    :return: classid_list, confidence_list, boxcoordinates_list
    """
    if os.path.exists(log_file):
        try:
            with open(log_file) as p:
                classid_list = []
                confidence_list = []
                boxcoordinates_list = []
                if log_file == "pyinfer_sample.txt":
                    keyStart = model_name + ":\n"
                    keyEnd = "end"
                    buff = p.read()
                    pat = re.compile(keyStart + "(.*?)" + keyEnd, re.S)
                    result = pat.findall(buff)
                    boxes_line = result[0].split("\n")
                else:
                    boxes_line = p.readlines()
                if len(boxes_line) == 0:
                    print("{} does not finded in pyinfer_sample.txt!".format(model_name))
                else:
                    if model_type == "det":
                        for line in boxes_line:
                            if "class_id" in line:
                                class_id = line.split(" ")[0]
                                confidence = line.split(" ")[1]
                                box_coordinates = line.split(" ")[2]
                                classid_list.append(class_id)
                                confidence_list.append(confidence)
                                boxcoordinates_list.append(box_coordinates)
                    else:
                        for line in boxes_line:
                            class_id = line.split(",")[1]
                            box_coordinates = line.split(",")[2:6]
                            confidence = line.split(",")[6]
                            classid_list.append("class_id:{}".format(class_id))
                            confidence_list.append("confidence:{}".format(confidence))
                            boxcoordinates_list.append(
                                "box:{},{},{},{}".format(
                                    box_coordinates[0], box_coordinates[1], box_coordinates[2], box_coordinates[3]
                                )
                            )
                return classid_list, confidence_list, boxcoordinates_list
        except:
            print("GRTTING data occur error from {}!".format(log_file))
            # return None, None, None
    else:
        print("{} does not exist ,CHECK PLEASE!!!".format(log_file))


def get_main_json_data(json_file, model_name):
    """
    :param json_file:
    :param model_name:
    :return: rectangular_list, confidence_list, kps_list
    """
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                s = json.load(f)
                kps_list = s[0][2][0]
                score = s[0][2][1]
                rect_list = s[0][1]
                num_kps = len(kps_list)
                confidence_list = []
                rectangular_list = []
                kpcoordinates_list = []
                for j in range(num_kps):
                    rectangular_list.append(
                        "rect:{},{},{},{}".format(rect_list[j][0], rect_list[j][1], rect_list[j][2], rect_list[j][3])
                    )
                    confidence_list.append("confidence:{}".format(score[j][0]))
                    for i in range(len(kps_list[j])):
                        kpcoordinates_list.append(
                            "kps:{},{},{}".format(kps_list[j][i][0], kps_list[j][i][1], kps_list[j][i][2])
                        )
                return rectangular_list, confidence_list, kpcoordinates_list
            except:
                print("{} get_main_json_data occur ERROR!!!".format(model_name))
    else:
        print("{} {} does not exist ,CHECK PLEASE!!!".format(model_name, json_file))


def diff_list(model_name, sample_list, run_list, type_list, confid_thr=0, box_thr=0.00):
    num_sample = len(sample_list)
    num_run = len(run_list)
    if num_sample == num_run:
        try:
            type_list_sign = "true"
            if type_list == "class_id":
                for i in range(num_sample):
                    if float(run_list[i].split(":")[1]) != float(sample_list[i].split(":")[1]):
                        type_list_sign = "false"
                        print(
                            "{} {}: RUN {} ，SAMPLE  {} ，CHECK PLEASE !".format(
                                model_name, type_list, run_list[i], sample_list[i]
                            )
                        )
            elif type_list == "confidence":
                for i in range(num_sample):
                    if (float(run_list[i].split(":")[1])) < (float(sample_list[i].split(":")[1]) * (1 - confid_thr)):
                        type_list_sign = "false"
                        print(
                            "{} {}: RUN {} ，SAMPLE  {} ，CHECK PLEASE !".format(
                                model_name, type_list, run_list[i], sample_list[i]
                            )
                        )
            else:
                for i in range(num_sample):
                    run_box = ast.literal_eval(run_list[i].split(":")[1])
                    sample_box = ast.literal_eval(sample_list[i].split(":")[1])
                    for inx, j in enumerate(run_box):
                        if j > sample_box[inx] * (1 + box_thr) or j < sample_box[inx] * (1 - box_thr):
                            type_list_sign = "false"
                            print(
                                "{} {}: RUN {} ，SAMPLE  {} ，CHECK PLEASE !".format(
                                    model_name, type_list, run_list[i], sample_list[i]
                                )
                            )
            if type_list_sign == "true":
                print("{} {} is SUCCESS".format(model_name, type_list))
        except:
            print("{} diff_list occur ERROR , CHECK LOG PLEASE!!!".format(model_name))
    else:
        print("{} num of predict box has diff , CHECK PLEASE !".format(model_name))


def run_pip(model_sample, model_run, model_type, model_name):
    type_list = ["class_id", "confidence", "box", "kps", "rect"]
    if model_type == "keypoint":
        rectangular_sample, confidence_sample, kpcoordinates_sample = get_main_json_data(
            model_sample, model_name=model_name
        )
        rectangular_run, confidence_run, kpcoordinates_run = get_main_json_data(model_run, model_name=model_name)
        diff_list(model_name, rectangular_sample, rectangular_run, type_list[4])
        diff_list(model_name, confidence_sample, confidence_run, type_list[1])
        diff_list(model_name, kpcoordinates_sample, kpcoordinates_run, type_list[3], box_thr=0.00)
    else:
        classid_sample, confidence_sample, boxcoordinates_sample = get_main_txt_data(
            model_sample, model_type, model_name=model_name
        )
        classid_run, confidence_run, boxcoordinates_run = get_main_txt_data(
            model_run, model_type, model_name=model_name
        )
        diff_list(model_name, classid_sample, classid_run, type_list[0])
        diff_list(model_name, confidence_sample, confidence_run, type_list[1])
        diff_list(model_name, boxcoordinates_sample, boxcoordinates_run, type_list[2], box_thr=0.00)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="keypoints or others")
    parser.add_argument("--sample_log", help="sample  log")
    parser.add_argument("--run_log", help="run model log")
    parser.add_argument("--model_name", help="model name")
    args = parser.parse_args()
    run_pip(args.sample_log, args.run_log, args.model_type, args.model_name)
