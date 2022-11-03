# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2022/9/6 3:46 PM
  * @brief  allure config
  *
  **************************************************************************/
"""

import os
import sys
import platform
import subprocess
import zipfile
import stat
import shutil
import wget


def make_tar(source_dir, output_filename):
    """
    打包文件夹
    """
    zipf = zipfile.ZipFile(output_filename, "w")
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    zipf.close()


def gen_allure_report():
    """
    install allure
    """
    exit_code, output = subprocess.getstatusoutput("allure --version")
    if exit_code == 0:
        print("allure version is:{}".format(output))
        allure_bin = "allure"
    else:
        sysstr = platform.system()
        os.system('wget -q https://xly-devops.bj.bcebos.com/tools/allure-2.19.0.zip')
        os.system('unzip allure-2.19.0.zip')
        allure_bin_f = "%s/allure-2.19.0/bin/allure" % (os.getcwd())
        st = os.stat(allure_bin_f)
        os.chmod(allure_bin_f, st.st_mode | stat.S_IEXEC)
        if sysstr == "Linux":
            # 验证完成分布式/mac/linux物理机已经完成java安装，暂时取消java安装
            # 没有root权限ln -s会失败
            allure_bin = "%s/allure-2.19.0/bin/allure" % (os.getcwd())
            # 额外check下是否安装java
            exit_code, output = subprocess.getstatusoutput("java -version")
            if exit_code == 0:
                print("java version is:{}".format(output))
            else:
                plat = platform.dist()[0].lower()
                retry_num = 0
                ret = 0
                while True:
                    retry_num += 1
                    if plat == "ubuntu":
                        ret = os.system("apt-get update;apt install -y openjdk-8-jdk >/dev/null")
                    elif plat == "centos":
                        ret = os.system("yum install java-1.8.0-openjdk-devel.x86_64 -y >/dev/null")
                    if ret == 0 or retry_num > 3:
                        break
        # windows和mac xly均已安装java
        elif sysstr == "Darwin":
            allure_bin = "%s/allure-2.19.0/bin/allure" % (os.getcwd())
        elif sysstr == "Windows":
            cmd = "set PATH=%s/allure-2.19.0/bin/allure;%%PATH%%" % (os.getcwd())
            os.system(cmd)
            # allure_bin = "allure"
            allure_bin = "%s/allure-2.19.0/bin/allure" % (os.getcwd())
    exit_code, output = subprocess.getstatusoutput("%s --version" % allure_bin)
    if exit_code == 0:
        print("allure version is:{}".format(output))
        cmd = "%s generate result -o report" % allure_bin
        ret = os.system(cmd)
        if ret:
            print("allure generate report failed")
        else:
            print("allure generate report sucess")
        REPORT_SERVER = "https://xly.bce.baidu.com/ipipe/ipipe-report"
        os.environ["REPORT_SERVER_USERNAME"] = "paddle"
        os.environ["REPORT_SERVER_PASSWORD"] = "915eedab953b6f51151f50eb"
        os.environ["REPORT_SERVER"] = REPORT_SERVER
        job_build_id = os.getenv("AGILE_JOB_BUILD_ID")
        if job_build_id:
            if os.path.exists("result"):
                make_tar("result", "result.tar")
                shutil.move("result.tar", "report")
            cmd = "curl -s {}/report/upload.sh | bash -s ./report {} report".format(REPORT_SERVER, job_build_id)
            print("upload cmd is {}".format(cmd))
            ret = os.system(cmd)
        return ret
    else:
        print("allure is not config correctly:{}, please config allure manually!".format(output))
        return 1


if __name__ == "__main__":
    gen_allure_report()
