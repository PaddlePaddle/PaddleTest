# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
config
"""

# fluid_sample 和 fluid 的层级关系对等（即fluid_sample = FluidDoc/doc/fluid）
# PADDLE_HOST = "http://180.76.141.178"
PADDLE_HOST = "https://www.paddlepaddle.org.cn"
# PADDLE_HOST = "http://sandbox.paddlepaddle.org.cn"
FLUIDDOC_PATH = "/Users/wangying28/Documents/paddle_github/FluidDoc/doc/fluid/"
PADDLE_PATH = "/Users/wangying28/Documents/paddle_github/Paddle/"

RESULT_SAMPLE_PATH = "paddle_sample/"
RESULT_SAMPLE_PATH_API_EN = RESULT_SAMPLE_PATH + "api/"
DIFF_HTML_PATH = RESULT_SAMPLE_PATH + "paddle_sample_diff/"
DIFF_HTML_HOST = "http://cp01-kgqa-plat.epc.baidu.com:8866/"
RESULT_EXCEL_FILE = RESULT_SAMPLE_PATH + "result.xlsx"

OWNER_API_PATH = "owner_api.xlsx"
OWNER_API_SHEET = "1.8版本"
OWNER_DOC_PATH = 'owner_doc.xlsx'
OWNER_DOC_SHEET = '工作表1'

ICAFE_USERNAME = 'wangying28'
ICAFE_PASSWORD = 'VVVGFxFUKEZpMgw1WB7evju6b59p6jk6rZi'
ICAFE_API_NEWCARD = 'http://icafe.baidu.com/api/v2/space/DLTP/issue/new'


MYSQL_HOSTNAME = '10.138.35.177'
MYSQL_PORT = 8859
MYSQL_USERNAME = 'metric'
MYSQL_PASSWORD = 'metric'
MYSQL_DATABASE = 'metric'
MYSQL_TABLE = 'paddle_deadlink'
