# encoding: utf-8
"""
生成windows,mac的model list文件
"""
import os
import yaml

cases_path = os.getcwd() + "/cases"
files = os.listdir(cases_path)

file_data_linux = ""
file_data_windows = ""
file_data_mac = ""

for file in files:
    with open(cases_path + "/" + file, "r", encoding="utf-8") as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
        if content["case"].get("windows"):
            if file_data_windows == "":
                file_data_windows += file
            else:
                file_data_windows += "\n" + file
        if content["case"].get("mac"):
            if file_data_mac == "":
                file_data_mac += file
            else:
                file_data_mac += "\n" + file

    if file_data_linux == "":
        file_data_linux += file
    else:
        file_data_linux += "\n" + file

with open("models_list_windows.txt", "w", encoding="utf-8") as f:
    f.write(file_data_windows + "\n")

with open("models_list_mac.txt", "w", encoding="utf-8") as f:
    f.write(file_data_mac + "\n")
