# encoding: utf-8
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file upload_report.py
  * @author wanghongtao
  * @date 2020/7/1 3:46 PM
  * @brief api test framework base class
  *
  **************************************************************************/
"""
import json
import requests
import os
def send(url):
    """
    以任务成功/失败的粒度上传报告
    """
    params = {
        "build_type_id": os.getenv('build_type_id'),
        "build_id": os.getenv('build_id'),
        "commit_id": os.getenv('build_commit_id'),
        "commit_time": os.getenv('build_commit_time'),
        "repo": os.getenv('build_repo_name'),
        "branch": os.getenv('build_repo_name'),
        "duration": 1,
        "exit_code": os.getenv('build_exit_code'),
        "status": os.getenv('build_status'),

        "case_detail": json.dumps([
            {
                "model_name": os.getenv('build_repo_name'),
                "kpi_name": os.getenv('models'),
                "kpi_status": os.getenv('build_status'),
                "kpi_base": 0,
                "kpi_value": 0,
                "threshold": 0,
                "ratio": 0
            }
        ])
    }
    res = requests.post(url, data=params)
    result =res.json()
    if result['code'] == 200 and result['message'] == 'success':
       print("ok")
    else:
       print('error')
    print(result)
if __name__ == "__main__":
    #url需配置在环境变量中，不允许上传到github；
    url = os.getenv('build_url')
    send(url)
