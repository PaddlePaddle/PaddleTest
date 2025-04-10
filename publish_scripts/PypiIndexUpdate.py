# !/usr/bin/env python3
"""获取PaddlePaddle安装包的所有依赖,并归档上传"""

import sys
import os
import platform
import subprocess
import multiprocessing
import argparse
import requests
from pathlib import Path
# 从Python SDK导入BOS配置管理模块以及安全认证模块
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient



def VersionVerification():
    """
    Python版本验证
    """
    python_version = platform.python_version()
    version_detail = sys.version_info
    version = str(version_detail[0]) + '.' + str(version_detail[1])
    env_version = os.getenv("PY_VERSION", None)


    if env_version is None:
        print(f"export PY_VERSION = {version}")
        os.environ["PY_VERSION"] = python_version

    elif env_version != version:
        raise ValueError(
            f"You have set the PY_VERSION environment variable to {env_version}, but "
            f"your current Python version is {version}, "
            f"Please keep them consistent."
        )
    return version


def download_requirement(paddle_whl_link=None, use_index=None):
    """
    下载PaddlePaddle及其依赖的Wheel包
    """
    cmd = [sys.executable, '-m', 'pip', 'download', paddle_whl_link, '--no-cache-dir']
    if use_index:
        cmd.extend(['-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])
    process = subprocess.Popen(cmd)
    process.wait()

    # 获取命令执行的输出
    # output = process.communicate()[0]
    # print(output.decode())

    # 获取命令执行的返回码
    return_code = process.returncode
    print("Return code:", return_code)


def create_dir_archive(target_path):
    """
    文件夹归档函数
    """
    # 遍历当前目录下的所有whl文件
    for filename in os.listdir(target_path):
        if filename.endswith('.whl'):
            print(filename)
            # 提取目录名，使用'-'作为分隔符
            dir_name = filename.split('-')[0]
            # 将'_'替换为'-'
            dir_name = dir_name.replace('_', '-')
            # 创建目录，如果目录已存在则忽略
            os.makedirs(dir_name, exist_ok=True)
            # 移动文件到相应目录
            os.replace(filename, os.path.join(dir_name, filename))
            # os.rename(filename, os.path.join(dir_name, filename))

def upload2bos(bucket_name, file_name):
    """
    Bos上传
    """
    # 设置BosClient的Host，Access Key ID和Secret Access Key
    bos_host = "bj.bcebos.com"
    access_key_id = os.getenv("AK")
    secret_access_key = os.getenv("SK")
    # 创建BceClientConfiguration
    config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
    bos_client = BosClient(config)

    # 使用 Path 类来标准化路径
    path = Path(file_name)
    # 转换为Linux格式
    linux_path = path.as_posix()
    object_key = linux_path

    result = bos_client.put_super_obejct_from_file(
        bucket_name, object_key, file_name, chunk_size=100, thread_num=multiprocessing.cpu_count()
    )
    if result:
        print("{} Upload success!".format(file_name))


def traverse_directory(path):
    """
    文件夹遍历，获取指定目录下的所有文件
    """
    all_files = []
    # 遍历路径下的所有文件层级
    for root, dirs, files in os.walk(path):
        for file in files:
            target_path = os.path.join(root, file)
            relative_path = os.path.relpath(target_path, path)
            all_files.append(relative_path)
    return all_files


def getHttpStatusCode(url):
    """
    链接状态判断
    """
    try:
        request = requests.get(url, timeout=5)
        httpStatusCode = request.status_code
        return httpStatusCode
    except requests.exceptions.HTTPError as e:
        return e
    except Exception as e:
        return e


def parse_description_url(des_url, py_version, packages_name='paddlepaddle_gpu', packages_version=None):
    """
    获取whl包的链接
    """
    whl_url_split = des_url.split('/')
    whl_url_pre = '/'.join(whl_url_split[:-2])
    post_whl_refer_dict = dict()
    # for item in ['3.8', '3.9', '3.10', '3.11', '3.12']:
    #     cur_ref = item.replace('.', '')
    #     post_whl_refer_dict[item] = f'cp{cur_ref}-cp{cur_ref}'
    cur_ref = py_version.replace('.', '')
    post_whl_refer_dict[py_version] = f'cp{cur_ref}-cp{cur_ref}'
    platform_str = ''
    platform_system = platform.system()
    machine = platform.machine()

    if platform_system == "Linux":
        if machine == "x86_64":
            platform_str = 'linux_x86_64'
        else:
            platform_str = 'linux_aarch64'
    elif platform_system == "Darwin":
        if machine == "x86_64":
            platform_str = 'macosx_10_9_x86_64'
        else:
            platform_str = 'macosx_11_0_arm64'
    else:
        platform_str = 'win_amd64'

    description_content = ''
    try:
        response = requests.get(des_url)
        response.raise_for_status()  # 检查请求是否成功
        description_content = response.text
        # return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        # return None

    # print(descriptsion_content)
    description_content_dict = dict()
    # 解析description中的文件
    for line in description_content.strip().split('\n'):
        key, value = line.strip().split(':')
        description_content_dict[key] = value

    # print(description_content_dict)
    commit_id = description_content_dict['commit_id']
    if packages_version is None:
        wheel_version = description_content_dict['wheel_version']
    else:
        wheel_version = packages_version
    cp_str = post_whl_refer_dict[py_version]
    whl_download_url = f"{whl_url_pre}/{commit_id}/{packages_name}-{wheel_version}-{cp_str}-{platform_str}.whl"

    return whl_download_url


def main():
    """
    主函数
    """
    # 验证Python版本是否正确
    py_version = VersionVerification()
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加一个位置参数，接受一个或多个整数值
    parser.add_argument('--packages_name', type=str, help='packages name(paddlepaddle)')
    parser.add_argument('--packages_version', dest='packages_version', default=None,
                    help='Specify the packages_version for stable version')
    parser.add_argument('--des_links', type=str, help='url links')
    parser.add_argument('--wheel_links', type=str, default=None, help='wheel package url links')
    parser.add_argument('--upload_path', dest='upload_path',
                    help='upload path relative to paddle-whl')
    parser.add_argument('--use_index', dest='use_index', default=None,
                    help='Specify the index used to download packages')
    # 解析命令行参数
    args = parser.parse_args()

    # 打印参数列表
    print("Provided Links:", args.des_links)
    print("Upload Path Relative To paddle-whl", args.upload_path)

    if args.wheel_links is not None:
        paddle_whl_link = args.wheel_links
    else:
        # 根据description文件和python版本拼接最新whl包的链接
        paddle_whl_link = parse_description_url(args.des_links, py_version, args.packages_name, args.packages_version)

    print("Current Wheel Link:", paddle_whl_link)
    # 获取当前工作目录
    original_path = os.getcwd()
    print("Current working directory:", original_path)


    # 要进入的目录路径
    target_directory = args.upload_path

    # 创建目录
    os.makedirs(target_directory, exist_ok=True)

    # 进入目录
    os.chdir(target_directory)

    # 再次获取当前工作目录
    print("New working directory:", os.getcwd())
    # 下载paddlepaddle的安装包以及依赖包
    download_requirement(paddle_whl_link, use_index=args.use_index)
    # 依赖包归档
    create_dir_archive(target_path=os.getcwd())
    # 回到初始目录
    os.chdir(original_path)
    print("New working directory:", os.getcwd())
    file_lists = traverse_directory(target_directory)
    # 判断文件是否存在，如果存在，则不需要重复上传，避免因为不必要的上传引入风险的可能
    for cur_file in file_lists:
        file_name = os.path.normpath(target_directory + cur_file)
        # 使用 Path 类来标准化路径
        path = Path(file_name)
        # 转换为Linux格式
        linux_path = path.as_posix()
        online_file_url = "https://paddle-whl.bj.bcebos.com/{}".format(linux_path)

        url_status = getHttpStatusCode(online_file_url)
        if url_status != 200:
            print("{} need upload".format(file_name))
            upload2bos('paddle-whl', file_name)
        else:
            print("file does not need to be uploaded:\n{}".format(file_name))
    


if __name__ == "__main__":
    main()
