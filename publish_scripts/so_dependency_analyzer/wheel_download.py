#!/usr/bin/env python3
import requests
import os

def build_wheel_url(base_url, python_version="3.10", is_cuda=True):
    """
    根据给定的基础URL、Python版本和是否使用CUDA生成PaddlePaddle的wheel文件下载链接。

    Args:
        base_url (str): 基础URL，用于生成下载链接。
        python_version (str, optional): Python版本，默认为"3.10"。
        is_cuda (bool, optional): 是否使用CUDA，默认为True。

    Returns:
        str: 生成的PaddlePaddle wheel文件下载链接。

    Raises:
        Exception: 如果下载描述文件失败，抛出异常。
        ValueError: 如果无法从描述文件中提取commit_id或wheel_version，抛出异常。

    """
    desc_url = f"{base_url}/latest/description.txt"
    response = requests.get(desc_url)
    if response.status_code != 200:
        raise Exception(f"❌ 下载失败: {desc_url}")
    
    commit_id = None
    wheel_version = None
    for line in response.text.splitlines():
        if line.startswith("commit_id:"):
            commit_id = line.split(":", 1)[1].strip()
        elif line.startswith("wheel_version:"):
            wheel_version = line.split(":", 1)[1].strip()
    
    if not commit_id or not wheel_version:
        raise ValueError("❗ 无法从 description.txt 中提取 commit_id 或 wheel_version")

    py_tag = f"cp{python_version.replace('.', '')}"
    if not is_cuda:
        package_name = "paddlepaddle"
    else:
        package_name = "paddlepaddle_gpu"
    if "ARM" in base_url:
        wheel_url = (
        f"{base_url}/{commit_id}/"
        f"{package_name}-{wheel_version}-{py_tag}-{py_tag}-linux_aarch64.whl"
        )
    else:
        wheel_url = (
            f"{base_url}/{commit_id}/"
            f"{package_name}-{wheel_version}-{py_tag}-{py_tag}-linux_x86_64.whl"
        )
    return wheel_url


def download_file(url, save_path):
    """
    从给定的URL下载文件并保存到指定的路径。

    Args:
        url (str): 要下载文件的URL。
        save_path (str): 文件保存的路径。

    Returns:
        None

    Raises:
        Exception: 如果下载失败，将引发异常。

    """
    # 确保文件夹存在
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"创建目录: {dir_name}")
    print(f"📥 正在下载: {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"❌ 下载失败: {url}")
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"✅ 保存成功: {save_path}")

if __name__ == "__main__":
    import argparse
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_version", type=str,default='3.10', help="Python version")
    parser.add_argument("--ce_task_name", type=str, help="CE task name")
    parser.add_argument("--is_cuda", type=str2bool, default=True, help="Whether to use CUDA")
    parser.add_argument("--output_dir", type=str, default=".", help="wheel package output directory")
    args = parser.parse_args()
    # parser.add_argument("base_url", type=str, help="Base URL of the PaddlePaddle wheel")
    # 示例调用
    base_url =f"https://paddle-qa.bj.bcebos.com/paddle-pipeline/{args.ce_task_name}"

    wheel_url = build_wheel_url(base_url, args.python_version, args.is_cuda)
    print("✅ 下载链接为：")
    print(wheel_url)
    description_url = f"{base_url}/latest/description.txt"
    description_save_path = os.path.join(args.output_dir, "description.txt")
    wheel_filename = os.path.basename(wheel_url)
    wheel_save_path = os.path.join(args.output_dir, wheel_filename)
    download_file(wheel_url, wheel_save_path)
    download_file(description_url, description_save_path)