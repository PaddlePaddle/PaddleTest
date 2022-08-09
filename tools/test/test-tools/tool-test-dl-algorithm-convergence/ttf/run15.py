#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import json
from flask import Flask
from flask_restful import Resource, Api
from flask import request

app = Flask(__name__)
api = Api(app)

# case_list = "mnist"

def run_cmd(cmd):
    import subprocess
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, universal_newlines=True)
    out, err = process.communicate()
    return out, process.returncode


@app.route("/tool-15")
def run():
    parameter_dict = request.get_json()
    cmd = "bash run_tools15.sh {} {}".format(parameter_dict['model_name'], parameter_dict['cards'])
    print(parameter_dict, cmd)

    out, ret = run_cmd(cmd)
    if ret != 0:
        result = {"status": 500, "msg": out, "result": "FAIL"}
    else:
        result = {"status": 200, "msg": out, "result": "PASS"}
    return json.dumps(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8315, debug=False)

# 正常响应（平均时延，单位毫秒）
#{"status": 200, "msg": "", "result": 9.1}
#失败响应
#{"status": 500, "msg": "error message", "result": "FAIL"}
