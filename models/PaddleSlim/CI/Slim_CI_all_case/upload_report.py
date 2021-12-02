# encoding: utf-8
import json
import requests
import os


def send(url):
    params = {
        "build_type": os.getenv('tc_build_type'),
        "task_name": os.getenv('tc_task_name'),
        "owner": os.getenv('tc_owner'),
        "build_id": os.getenv('tc_build_id'),
        "build_number": os.getenv('tc_build_number'),
        "commit_id": os.getenv('tc_commit_id'),
        "repo": os.getenv('tc_repo_name'),
        "branch": os.getenv('tc_repo_branch'),
        "status": os.getenv('tc_status'),
        "exit_code": os.getenv('tc_exit_code'),
        "create_time": os.getenv('tc_create_time'),
        "duration": 1,
        "case_detail": json.dumps([
            {
                "model_name": os.getenv('tc_task_name'),
                "kpi_name": os.getenv('tc_kpi_name'),
                "kpi_status": os.getenv('tc_status'),
                "kpi_base": 0,
                "kpi_value": 0,
                "threshold": 0,
                "ratio": 0
            }
        ])
    }

    print(params)
    res = requests.post(url, data=params)
    return res


if __name__ == "__main__":
    #url需配置在环境变量中，不允许上传到github；
    url = os.getenv('tc_url')
    print(send(url))