# encoding: utf-8
"""
binary search
"""
import logging
import os
import subprocess
import argparse
import tarfile

# import logging
import re
import paddle
import wget
import git


def parse_args():
    """
    接收命令传入的参数
    """
    parser = argparse.ArgumentParser("Tool for binary_search")
    parser.add_argument("--bad_commit", help="出现问题的commit", type=str, default="HEAD")
    parser.add_argument("--good_commit", help="上一次没有问题的commit", type=str, default=None)
    args = parser.parse_args()
    return args


def get_cmd_output(cmd):
    """
    获取cmd的输出结果
    """
    print("cmd:{}".format(cmd))
    result = subprocess.getstatusoutput(cmd)
    output = result[1]
    print("cmd output:\n{}".format(output))
    commit_id = re.findall(r"\[(.{40})\]", output)

    print("commit_id_list:{}".format(commit_id))
    if commit_id:
        commit_id = commit_id[0]
        print("commit_id:{}".format(commit_id))
        cmd = (
            "python -m pip uninstall -y paddlepaddle-gpu;\
python -m pip install -U https://paddle-qa.bj.bcebos.com/paddle-pipeline/\
Develop-GpuAll-LinuxCentos-Gcc82-Cuda112-Trtoff-Py38-Compile/%s/\
paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl -i https://mirror.baidu.com/pypi/simple"
            % (commit_id)
        )
        os.system(cmd)
        print("installed paddle commit_id:")
        os.system('python -c "import paddle; print(paddle.version.commit)"')

    return output


def run(args):
    """
    run
    """
    bad_commit = args.bad_commit
    good_commit = args.good_commit
    # os.system('python -m pip install paddlepaddle-gpu==2.4.2.post112\
    #  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html\
    #  -i https://mirror.baidu.com/pypi/simple')
    # paddle_repo = git.Repo.clone_from(url='https://github.com/PaddlePaddle/Paddle.git',\
    # to_path='Paddle', branch='develop')
    if not os.path.exists("Paddle"):
        wget.download("https://xly-devops.bj.bcebos.com/PaddleTest/Paddle/Paddle.tar.gz")
        tf = tarfile.open("Paddle.tar.gz")
        tf.extractall(os.getcwd())

    cmd = "cd /paddle/Paddle; git bisect start %s %s" % (bad_commit, good_commit)
    print("cmd:{}".format(cmd))
    output = get_cmd_output(cmd)

    the_first_bad_commit_flag = False
    while not the_first_bad_commit_flag:
        result = subprocess.getstatusoutput("cd /paddle/ocr_mt/PaddleTest/models_restruct/Paddle3D; bash run_3d.sh")
        output = result[1]
        print(output)

        if "failed" in output:
            print("***test case failed!***")
            output = get_cmd_output("cd /paddle/Paddle; git bisect bad")
        else:
            print("***test case success!***")
            output = get_cmd_output("cd /paddle/Paddle; git bisect good")

        if "the first bad commit" in output:
            the_first_bad_commit_flag = True
        else:
            the_first_bad_commit_flag = False
    print("the_first_bad_commit is:{}".format(output))


if __name__ == "__main__":
    """
    main
    """
    args = parse_args()
    run(args)
