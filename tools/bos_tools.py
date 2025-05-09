#!/usr/bin/env python
# coding=utf-8
"""
upload script
"""
import os
import multiprocessing
import argparse
import time

# 从Python SDK导入BOS配置管理模块以及安全认证模块
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient


# 设置BosClient的Host，Access Key ID和Secret Access Key
bos_host = "bj.bcebos.com"
access_key_id = os.getenv("AK")
secret_access_key = os.getenv("SK")
# 创建BceClientConfiguration
config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
bos_client = BosClient(config)

def bos_upload(bucket_name, object_key, file_name):
    result = bos_client.put_super_obejct_from_file(
        bucket_name, object_key, file_name, chunk_size=100, thread_num=multiprocessing.cpu_count()
    )
    if result:
        print("Upload success!")

def bos_download_url(bucket_name, object_key):
    # 建议使用0.9.29版本的bce-python-sdk
    print("Please ensure you are using bce-python-sdk version 0.9.29 or later.")
    bos_host = bucket_name + ".bj.bcebos.com"
    config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
    bos_client = BosClient(config)

    timestamp = int(time.time())
    url_bytes = bos_client.generate_pre_signed_url(bucket_name, object_key, timestamp,
                                               expiration_in_seconds=1800)
    url = url_bytes.decode('utf-8')
    return url

def bos_download(bucket_name, object_key):
    download_file_name = os.path.basename(object_key)
    result = bos_client.get_object_to_file(bucket_name, object_key, download_file_name)
    if result:
        print("Download success!")


# 用法说明函数
def print_usage():
    print("""
    用法:
    python bos_tools.py [操作] [对象键] [存储桶名称] [操作类型]
    
    参数:
    操作        上传或下载文件到存储桶，支持两种操作：upload, download 或 download_url。
    对象键      需要上传或下载的文件路径。
    存储桶名称  需要操作的存储桶名称（如: paddle-qa）。
    操作类型    可选参数，指定操作类型，默认为 'upload'，可以选择 'download' 进行下载。
    
    示例:
    1. 上传文件：
       python bos_tools.py my_file_upload.txt paddle-qa/upload_folder upload
       
    2. 下载文件：
       python bos_tools.py DistributedCE/PGLBOX/download_sample paddle-qa download

    3. 获取下载链接(私有bucket默认为半小时有效):
       python bos_tools.py DistributedCE/PGLBOX/download_sample paddle-qa download_url
    """)




if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="上传或下载文件到存储桶")

    # 添加参数
    parser.add_argument("object_key", help="对象键（文件名）")
    parser.add_argument("bucket_name", help="存储桶名称")
    parser.add_argument("operation", nargs="?", choices=["upload", "download", "download_url"], default="upload",
                        help="操作类型(upload 或 download)，默认 upload")

    # 解析命令行参数
    args = parser.parse_args()

    # 打印用法说明
    if len(vars(args)) == 0:
        print_usage()
        exit()

    # 判断操作类型
    if args.operation == "upload":
        bucket_name = args.bucket_name.split('/')[0]
        file2upload = args.object_key
        object_key = '/'.join(args.bucket_name.split('/')[1:]) + '/' + file2upload
        print(f"正在上传 {file2upload} 到存储桶 {bucket_name}, 路径为{object_key} ...")
        # 在这里调用你的上传函数，例如 upload_file(args.bucket_name, args.object_key)
        bos_upload(bucket_name, object_key, file2upload)
    elif args.operation == "download":
        print(f"正在从存储桶 {args.bucket_name} 下载 {args.object_key} ...")
        # 在这里调用你的下载函数，例如 download_file(args.bucket_name, args.object_key)
        # bos_download(args.bucket_name, args.object_key)
        bos_download(args.bucket_name, args.object_key)
    elif args.operation == "download_url":
        print(f"获取存储桶 {args.bucket_name}中{args.object_key}的下载链接 ...")
        url = bos_download_url(args.bucket_name, args.object_key)
        print(url)