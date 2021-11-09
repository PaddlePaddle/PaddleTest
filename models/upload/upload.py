# encoding: utf-8
"""
上传到bos
"""
#导入BOS相关模块
import os
import sys
from baidubce import exception
from baidubce.services import bos
from baidubce.services.bos import canned_acl
from baidubce.services.bos.bos_client import BosClient
from bos_conf import config

#新建BosClient
bos_client = BosClient(config)
# 将指定文件夹下的文件上传上去
bucket_name = os.path.join('paddle-qa', sys.argv[2])
object_key_format = '{file_name}'

def upload_file(file_path):
    """
    object_key: bos上生成的路径
    file_path:要上传的文件夹绝对路径
    """
    file_name=os.path.basename(file_path)
    object_key = object_key_format.format(file_name=file_name)
    bos_client.put_object_from_file(bucket_name, object_key, file_path)

if __name__ == "__main__":
    upload_file(sys.argv[1])
