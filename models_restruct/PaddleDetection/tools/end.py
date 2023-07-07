# encoding: utf-8
"""
执行case后执行的文件
"""
import os

logger = logging.getLogger("ce")

class PaddleDetection_End(object):
    """
    case执行结束后
    """

    def __init__(self):
        """
        初始化
        """
        self.reponame = os.environ["reponame"]
        self.model = os.environ["model"]
        self.qa_model_name = os.environ["qa_yaml_name"]
        self.log_dir = "logs"
        self.log_name = "train_prim_single.log"
        self.log_path = os.path.join(os.getcwd(), self.log_dir, self.reponame, self.qa_model_name, self.log_name)
        logger.info("log_path:{}".format(self.log_path))
        self.qa_model_name_base = ""
        self.prim_log_path = ""
        self.standard_log_path = ""

    def build_end(self):
        """
        执行准备过程
        """
        # kill遗留程序
        logger.info("PID before is {}".format(os.system(f"ps aux| grep '{self.qa_model_name}'| grep -v 'main.py'")))
        cmd_kill = os.system(
            f"ps aux | grep '{self.qa_model_name}' | grep -v 'main.py' | awk '{{print $2}}' | xargs kill -9"
        )
        # cmd_kill = os.system(f"pkill -f '{self.qa_model_name}'| grep -v 'main.py'") #这样会先执行kill再执行grep没有生效
        logger.info("PID after is {}".format(os.system(f"ps aux| grep '{self.qa_model_name}'| grep -v 'main.py'")))
        # 异常256
        logger.info("cmd_kill is {}".format(cmd_kill))
        return 0


def run():
    """
    执行入口
    """
    model = PaddleDetection_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
