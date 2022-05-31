import json
import requests
import paddle
import os

def send(url):
    case_result = []
    with open('result', 'r') as f:
        for line in f:
            line = line.strip('\n')
            sub_str = line.split(',')
            case_result.append({"model_name": sub_str[0],
                                "step_name": sub_str[1], 
                                "kpi_name": "loss",
                                "kpi_status": sub_str[2],
                                "kpi_base": 0,
                                "kpi_value": 0,
                                "threshold": 0,
                                "ratio": 0})     
    params = {
         "build_type_id": os.getenv('build_type_id'),
         "build_id": os.getenv('build_id'),
         "commit_id": paddle.version.commit,
         #"commit_time": "",
         "repo": os.getenv('repo'),
         "branch": os.getenv('branch'),
         "status": os.getenv('status'),
         "exit_code": os.getenv('exit_code'),
         "duration": 200,
         "case_detail": json.dumps(case_result)
    }
    res = requests.post(url, data=params)
    result = res.json()
    print('result:{}'.format(result))
    if result['code'] == 200 and result['message'] == 'success':
       print("ok")
    else:
       print('error')

if __name__ == "__main__":
    url = "http://wangying28-2020-004-3.bcc-bdbl.baidu.com:8999/api/case"
    send(url)
