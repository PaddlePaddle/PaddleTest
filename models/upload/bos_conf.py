#!/usr/bin/env python
# encoding: utf-8
"""
bos_config
"""
# 导入Python标准日志模块
import logging
import os

# 从Python SDK导入BOS配置管理模块以及安全认证模块
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials

bos_host = "bj.bcebos.com"
access_key_id = os.environ.get("access_key_id")
secret_access_key = os.environ.get("secret_access_key")

logger = logging.getLogger("baidubce.http.bce_http_client")
fh = logging.FileHandler("sample.log")
fh.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(fh)

config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
