# encoding: utf-8
"""
默认环境准备
"""
import os
import time
import wget
import tarfile

class Model_Build(object):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        self.paddle_whl = args.paddle_whl
        self.reponame = args.reponame
        self.get_repo = args.get_repo
        self.branch = args.branch
        self.set_cuda = args.set_cuda
        self.dataset_org = args.dataset_org
        self.dataset_target = args.dataset_target

    def build_env_whl(self):
        """
        获取whl包和模型repo，安装whl包
        """
        #安装paddle
        os.system("python -m pip install --upgrade pip")
        if str(self.paddle_whl) == "None":    #判断是否需要安装包
            print("use org paddle do not install")
        else:
            os.system("python -m pip uninstall -y paddlepaddle")
            os.system("python -m pip uninstall -y paddlepaddle-gpu")
            cmd_return = os.system("python -m pip install {}".format(self.paddle_whl))
            if cmd_return:
                print("whl {} install failed".format(self.paddle_whl))
                #TODO 退出码和后续任务标注对齐
                return 1 
        
        #clone模型库，支持如果之前clone好就跳过
        if os.path.exists(self.reponame):
            print("already have {}, if need new please del it".format(self.reponame))
        else:
            print("####get_repo", self.get_repo)
            if str(self.get_repo) == "clone": #clone库
                print("clone {}".format(self.reponame))
                github_repo = "https://github.com/PaddlePaddle/{}.git".format(self.reponame)
                os.system("git clone -b {} {}" \
                        .format(self.branch, github_repo)) 
            else: #用天级打包的gz包
                # TODO 依赖wget tarfile包，需要预先下载
                print("wget {}".format(self.reponame))
                wget.download("https://xly-devops.bj.bcebos.com/PaddleTest/{}.tar.gz" \
                    .format(self.reponame))
                tf = tarfile.open("{}.tar.gz".format(self.reponame))
                tf.extractall(os.getcwd())
                os.remove("{}.tar.gz".format(self.reponame))
            if os.path.exists(self.reponame) is False:
                #TODO 退出码和后续任务标注对齐
                print("repo {} clone failed".format(self.reponame))
                return 1
        
        #安装依赖包
        #TODO 明确是不是所有repo都是requirmetns ，明确是否需要额外修改其它信息或者安装其它依赖
        cmd_return = os.system("cd {} && python -m pip install -r requirements.txt".format(self.reponame))
        if cmd_return:
            print("repo {} pip install requirements failed".format(self.reponame))
            return 1
        #TODO 返回框架信息，模型repo信息
        return 0 
    
    def build_env_param(self):
        """
        设置需要的环境变量，通过main调用只在build，后续进程系统变量可生效
        """
        #设置显卡号
        os.environ['CUDA_VISIBLE_DEVICES'] = self.set_cuda
        print("#### set CUDA_VISIBLE_DEVICES as {}".format(self.set_cuda))
        return 0
    
    def build_env_data(self):
        """
        设置数据集合
        dataset：
            None 不做处理
        """
        if str(self.dataset_org) == "None":
            print("do not link dataset")
        else:
            #target如果有先命名为back
            if os.path.exists(os.path.join(self.reponame, self.dataset_target)):
                print("already have {} will mv {} to {}".format(self.dataset_target, self.dataset_target, \
                    self.dataset_target + "_" + str(int(time.time()))))
                os.rename(os.path.join(self.reponame, self.dataset_target), \
                    os.path.join(self.reponame, self.dataset_target + "_" + str(int(time.time()))))
            cmd_return = os.symlink(self.dataset_org, os.path.join(self.reponame, self.dataset_target))
            if cmd_return:
                #TODO 退出码和后续任务标注对齐
                print("从 {} 软链到 {} 失败".format(self.dataset_org, self.dataset_target))
                return 1
        return 0

    def build_env(self):
        """
        搭建环境入口
        """
        ret = 0
        ret = self.build_env_whl()
        if ret:
            print("build env whl failed")
            return ret
        ret = self.build_env_param()
        if ret:
            print("build env params failed")
            return ret
        ret = self.build_env_data()
        if ret:
            print("build env data failed")
            return ret
        return ret
