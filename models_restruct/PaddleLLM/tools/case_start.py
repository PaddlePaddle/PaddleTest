# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
"""
import os
import logging
import re

logger = logging.getLogger("ce")


class PaddleNLP_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.reponame = os.environ["reponame"]
        self.mode = os.environ["mode"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]

    def build_prepare(self):
        """
        执行准备过程
        """
        if str(os.getenv("SOT_EXPORT_FLAG")) == "True":
            os.environ["SOT_EXPORT"] = f"Layer_cases/{self.qa_yaml_name}_{self.case_name}_{self.case_step}"
            logger.info("set org SOT_EXPORT as {}".format(os.getenv("SOT_EXPORT")))


class PaddleLLM(object):
    """
    起始hook
    """
    def __init__(self):
        """
        初试化
        """
        self.system = os.getenv("system")
        self.reponame = os.getenv("reponame")
        self.model_name = os.getenv("model_name")
        self.case_step = os.getenv("case_step")
        self.case_name = os.getenv("case_name")
        self.branch = os.getenv("branch")

    def build_start(self):
            """
            执行准备过程
            """
            #case 执行前清理显卡
            def kill_nvidia_processes():
                # 找出并终止使用 NVIDIA 显卡的进程
                try:
                    fuser_output = subprocess.check_output(
                        "fuser -v /dev/nvidia* 2>/dev/null", shell=True
                    ).decode("utf-8")
                    pids = [int(word) for word in fuser_output.split() if word.isdigit()]
                    if pids:
                        subprocess.run(["kill", "-9"] + [str(pid) for pid in pids])
                    return True
                except subprocess.CalledProcessError:
                    return False

            def kill_pyxes_process():
                # 找出并终止包含 'pyxes' 的进程
                try:
                    subprocess.run(["pkill", "-f", "pyxes"])
                    return True
                except subprocess.CalledProcessError:
                    return False

                # 执行两个清理函数，并处理执行结果
            if kill_nvidia_processes() and kill_pyxes_process():
                print("显存清理成功")
            else:
                print("显存清理PASS")
            
            if os.getenv("RUN_AUTOTUNER") == "1":
                log_dir = os.path.join("./logs/" + self.reponame + "_" + self.branch + "_" + \
                                       self.system, self.model_name, self.case_step + "_" + self.case_name)
                logger.info(f"{log_dir} is created")    
                os.makedirs(log_dir, exist_ok=True)
            
            def get_message_queue_ids():
                """
                获取系统中所有的消息队列 ID (msqid)。
                :return: 包含 msqid 的列表
                """
                try:
                    # 调用 `ipcs -q` 命令并捕获输出
                    result = subprocess.run(["ipcs", "-q"], stdout=subprocess.PIPE, text=True)
                    lines = result.stdout.splitlines()

                    # 跳过标题行，从第 4 行开始解析 msqid
                    msqids = [line.split()[1] for line in lines[3:] if len(line.split()) > 1]
                    return msqids
                except Exception as e:
                    print(f"Error fetching message queue IDs: {e}")
                return []

            def delete_message_queue(msqid):
                """
                删除指定的消息队列。
                :param msqid: 消息队列 ID
                """
                try:
                    subprocess.run(["ipcrm", "-q", msqid], check=True)
                    print(f"Successfully deleted message queue with msqid: {msqid}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to delete message queue with msqid: {msqid}. Error: {e}")
                except Exception as e:
                    print(f"Unexpected error while deleting msqid {msqid}: {e}")


            print("Fetching all message queue IDs...")
            msqids = get_message_queue_ids()

            if not msqids:
                print("No message queues found.")
                return

            print(f"Found message queues: {msqids}")

            for msqid in msqids:
                print(f"Deleting message queue with msqid: {msqid}")
                delete_message_queue(msqid)

            print("All message queues processed.")

def run():
    """
    执行入口
    """
    if os.getenv("reponame") == "PaddleLLM":
        model = PaddleLLM()
        model.build_start()
    else:
        model = ErnieCaseStart()
        model.build_start()
    return 0



if __name__ == "__main__":
    run()
