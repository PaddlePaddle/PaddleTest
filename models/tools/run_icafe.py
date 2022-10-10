#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2022/10/10 3:46 PM
  * @brief  create icafe
  *
  **************************************************************************/
"""


import sys
import urllib
import urllib.request
import json
import time
import datetime

icafe_api = "http://icafeapi.baidu-int.com/api/v2"
icafe_debug = "http://icafebeta.baidu.com/api/v2"
api_entry = icafe_api

space_id = "DLTP"


def create_issue(
    title,
    detail,
    repo,
    password,
    AGILE_PULL_ID,
    xly_link,
    type="Bug",
    emailto=[
        "jiaxiao01@baidu.com",
    ],
    owner="jiaxiao01",
):
    """
    create_issue
    """
    print(" ------------- 1. cretae issue-----------------")
    url = "%s/space/%s/issue/new" % (api_entry, space_id)
    print(url)
    i_headers = {"Content-Type": "application/json"}
    values = {
        "username": "jiaxiao01",
        "password": "VVVElLuWkVHSZb1e1bgS%2Fm68A%3D%3D",
        "issues": [
            {
                "title": title,
                "detail": detail,
                "type": type,
                "fields": {
                    "负责人": owner,
                    "所属计划": "QA反馈/【QA反馈】模型bug&&易用性问题",
                    "负责人所属团队": "QA团队",
                    "需求来源": "QA团队",
                    "QA负责人": "jiaxiao01",
                    "优先级": "P1-严重问题 High",
                    "issue链接": xly_link,
                    "PR链接": "https://github.com/PaddlePaddle/" + repo + "/pull/" + AGILE_PULL_ID,
                    "bug发现方式": "模型CI任务",
                    "repo": repo,
                },
                "parent": 58414,
                "notifyEmails": emailto,
                "creator": "jiaxiao01",
            }
        ],
    }
    data = json.dumps(values).encode("UTF8")
    print(data)
    req = urllib.request.Request(url, data, headers=i_headers)
    response = urllib.request.urlopen(req)
    the_page = response.read()
    print(the_page)
    # read response
    response_json = json.loads(the_page)
    if response_json.get("status", "") == 200:
        print("Success!!!")
    else:
        print("Fail, details: %s") % response_json.get("message", "")
        exit(1)


if __name__ == "__main__":
    """
    create_issue
    """
    repo = sys.argv[1]
    category = sys.argv[2]
    keyword = sys.argv[3]
    password = sys.argv[4]
    AGILE_PIPELINE_BUILD_ID = sys.argv[5]
    AGILE_JOB_BUILD_ID = sys.argv[6]
    AGILE_PULL_ID = sys.argv[7]
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    xly_link = (
        "https://xly.bce.baidu.com/paddlepaddle/"
        + keyword
        + "/newipipe/detail/"
        + AGILE_PIPELINE_BUILD_ID
        + "/job/"
        + AGILE_JOB_BUILD_ID
    )
    title = "[auto CI]" + " " + repo + "_" + category + " " + date
    detail = xly_link
    create_issue(title, detail, repo, password, AGILE_PULL_ID, xly_link)
