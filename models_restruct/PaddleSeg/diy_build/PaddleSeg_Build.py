# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import logging
import tarfile
import argparse
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleSeg_Build(Model_Build):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        self.paddle_whl = args.paddle_whl
        self.get_repo = args.get_repo
        self.branch = args.branch
        self.system = args.system
        self.set_cuda = args.set_cuda
        self.dataset_org = args.dataset_org
        self.dataset_target = args.dataset_target

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
        self.reponame = args.reponame
        self.models_list = args.models_list
        self.models_file = args.models_file
        self.clas_model_list = []
        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" in line:
                    self.clas_model_list.append(line.strip().replace("-", "/"))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" in line:
                        self.clas_model_list.append(line.strip().replace("-", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" in file_name:
                    self.clas_model_list.append(file_name.strip().replace("-", "/"))

    def value_in_modellist(self, value=None):
        """
        判断字段是否存在model_list
        """
        for line in self.clas_model_list:
            if value in line:
                return 1
        return 0

    def unzip_file(self, zip_src, dst_dir):
        import zipfile

        r = zipfile.is_zipfile(zip_src)
        if r:
            fz = zipfile.ZipFile(zip_src, 'r')
            for file in fz.namelist():
                fz.extract(file, dst_dir)
        else:
            print('This is not zip')

    def download_data(self, value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleSeg

        tar_name = value.split("/")[-1]
        # if os.path.exists(tar_name) and os.path.exists(tar_name.replace(".tar", "")):
        # 有end回收数据，只判断文件夹
        if value.strip().endswith('.tar'):
            if os.path.exists(tar_name.replace(".tar", "")):
                logger.info("#### already download {}".format(tar_name))
            else:
                try:
                    logger.info("#### start download {}".format(tar_name))
                    wget.download(value.replace(" ", ""))
                    logger.info("#### end download {}".format(tar_name))
                    tf = tarfile.open(tar_name)
                    tf.extractall(os.getcwd())
                except:
                    logger.info("#### prepare download failed {} failed".format(tar_name))
        elif value.strip().endswith('.tar.gz'):
            if os.path.exists(tar_name.replace(".tar.gz", "")):
                logger.info("#### already download {}".format(tar_name))
            else:
                try:
                    logger.info("#### start download {}".format(tar_name))
                    wget.download(value.replace(" ", ""))
                    logger.info("#### end download {}".format(tar_name))
                    tf = tarfile.open(tar_name)
                    tf.extractall(os.getcwd())
                except:
                    logger.info("#### prepare download failed {} failed".format(tar_name))
        elif value.strip().endswith('.tgz'):
            if os.path.exists(tar_name.replace(".tgz", "")):
                logger.info("#### already download {}".format(tar_name))
            else:
                try:
                    logger.info("#### start download {}".format(tar_name))
                    wget.download(value.replace(" ", ""))
                    logger.info("#### end download {}".format(tar_name))
                    tf = tarfile.open(tar_name)
                    tf.extractall(os.getcwd())
                except:
                    logger.info("#### prepare download failed {} failed".format(tar_name))
        elif value.strip().endswith('.zip'):
            if os.path.exists(tar_name.replace(".zip", "")):
                logger.info("#### already download {}".format(tar_name))
            else:
                try:
                    logger.info("#### start download {}".format(tar_name))
                    wget.download(value.replace(" ", ""))
                    logger.info("#### end download {}".format(tar_name))
                    self.unzip_file(tar_name, './') 
                except Exception as e:
                    logger.info("#### prepare download failed {} failed".format(tar_name))
                    logger.info("#### {}".format(e))
        else:
            if os.path.exists(tar_name):
                logger.info("#### already download {}".format(tar_name))
            else:
                try:
                    logger.info("#### start download {}".format(tar_name))
                    wget.download(value.replace(" ", ""))
                    logger.info("#### end download {}".format(tar_name))
                except:
                    logger.info("#### prepare download failed {} failed".format(tar_name))
     
        return 0

    def get_image_name(self, value=None, label=None):
        """
        获取数据集名称
        """
        with open(os.path.join(self.REPO_PATH, value), "r", encoding="utf-8") as y:
            cfg = yaml.full_load(y)
            image_name = cfg["DataLoader"]["Train"]["dataset"][label]
            # logger.info('####image_name: {}'.format(image_name))
            image_name = image_name.split("dataset")[1]
            # logger.info('####image_name: {}'.format(image_name))
            image_name = image_name.split("/")[1]
            # logger.info('####image_name: {}'.format(image_name))
            image_name = image_name.replace('"', "")
            # logger.info('####image_name: {}'.format(image_name))
        return image_name

    def change_yaml_batch_size(self, data_json):
        """
        递归使所有batch_size,默认除以3
        """
        if isinstance(data_json, dict):
            for key, val in data_json.items():
                if key == "batch_size" and "@" not in str(val):
                    data_json[key] = str(int(np.ceil(float(val) / 3))) + "  #@"
                if isinstance(data_json[key], dict):
                    self.change_yaml_batch_size(data_json[key])
        return data_json

    def build_yaml(self):
        """
        更改RD原始的yaml文件
        demo: PaddleSeg/ppcls/configs/ImageNet/ResNet/ResNet50.yaml
        """
        if os.path.exists(self.reponame):
            for line in self.clas_model_list:
                with open(os.path.join(self.REPO_PATH, line), "r") as f:
                    content = yaml.load(f, Loader=yaml.FullLoader)

                # 改变 batch_size
                if "PKSampler" in str(content) or "DistributedRandomIdentitySampler" in str(content):
                    logger.info("#### do not change batch_size in {}".format(line))
                else:
                    content_new = self.change_yaml_batch_size(content)  # 写在with里面不能够全部生效
                    with open(os.path.join(self.REPO_PATH, line), "w") as f:
                        yaml.dump(content_new, f, sort_keys=False)

                # 改变 GeneralRecognition 依赖的数据集
                if "GeneralRecognition" in line:
                    content["DataLoader"]["Train"]["dataset"]["image_root"] = "./dataset/Inshop/"
                    content["DataLoader"]["Train"]["dataset"]["cls_label_path"] = "./dataset/Inshop/train_list.txt"
                    content["DataLoader"]["Eval"]["Query"]["dataset"]["image_root"] = "./dataset/iCartoonFace/"
                    content["DataLoader"]["Eval"]["Gallery"]["dataset"]["image_root"] = "./dataset/iCartoonFace/"
                    content["DataLoader"]["Eval"]["Query"]["dataset"][
                        "cls_label_path"
                    ] = "./dataset/iCartoonFace/gallery.txt"
                    content["DataLoader"]["Eval"]["Gallery"]["dataset"][
                        "cls_label_path"
                    ] = "./dataset/iCartoonFace/gallery.txt"
                    with open(os.path.join(self.REPO_PATH, line), "w") as f:
                        yaml.dump(content, f)
        else:
            logger.info("check you {} path".format(self.reponame))
        return 0

    def build_dataset(self):
        """
        自定义下载数据集
        """
        if os.path.exists(self.reponame):
            # 下载模型
            path_now = os.getcwd()
            os.chdir(self.reponame)
            #onnx
            if  os.path.exists("pp_liteseg_stdc1_fix_shape"):
                logger.info("#### already download pp_liteseg_stdc1_fix_shape")
            else:
                logger.info("#### start download pp_liteseg_stdc1_fix_shape")
                self.download_data(
                    value="https://paddleseg.bj.bcebos.com/tipc/infer_models/pp_liteseg_stdc1_fix_shape.zip"
                )
                logger.info("#### end download pp_liteseg_stdc1_fix_shape")
            #ptq
            if  os.path.exists("pp_liteseg_infer_model"):
                logger.info("#### already download pp_liteseg_infer_model")
            else:
                logger.info("#### start download pp_liteseg_infer_model")
                self.download_data(
                    value="https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz"
                )
                logger.info("#### end download pp_liteseg_infer_model")
            os.mkdir("./test_tipc/data/")
            os.chdir("./test_tipc/data/")
            logger.info("#### start download pp_liteseg_infer_data: cityscapes")
            self.download_data(
                value="https://paddleseg.bj.bcebos.com/tipc/data/cityscapes_20imgs.tar"
            )
            os.rename("cityscapes_20imgs", "cityscapes")
            logger.info("#### end download pp_liteseg_infer_data: cityscapes")
            #pact
            import shutil
            shutil.copy(path_now + "/" + self.reponame + "/test_tipc/docs/cityscapes_val_5.list", "cityscapes/")
            #cppinfer
            #serving_python
            os.chdir(path_now)
            os.chdir(self.reponame)
            if  os.path.exists("pp_liteseg_infer_model"):
                logger.info("#### already download pp_liteseg_infer_model")
            else:
                logger.info("#### start download pp_liteseg_infer_model")
                self.download_data(
                    value="https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz"
                )
                logger.info("#### end download pp_liteseg_infer_model")
            if  os.path.exists("cityscapes_small.png"):
                logger.info("#### already download cityscapes_small.png")
            else:
                logger.info("#### start download cityscapes_small.png")
                self.download_data(
                    value="https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_small.png"
                )
                logger.info("#### end download cityscapes_small.png")
 

            os.chdir(path_now)
        return 0


    def build_paddleseg(self):
        """
        安装依赖包
        """
        # 固定随机量需要，默认打开
        os.environ["FLAGS_cudnn_deterministic"] = "True"
        logger.info("#### set FLAGS_cudnn_deterministic as {}".format(os.environ["FLAGS_cudnn_deterministic"]))

        path_now = os.getcwd()
        os.chdir("PaddleSeg")  # 执行setup要先切到路径下面

        cmd_return = os.system("python -m pip install --retries 10 -r requirements.txt")
        if cmd_return:
            logger.info("repo {} python -m pip install -r requirements.txt failed".format(self.reponame))
        cmd_return = os.system("pip install -e .")
        if cmd_return:
            logger.info("repo {} python -m pip install -e . failed".format(self.reponame))
        cmd_return = os.system("python -m pip install --retries 10 scikit-image")
        if cmd_return:
            logger.info("repo {} python -m pip install scikit-image failed".format(self.reponame))
        cmd_return = os.system("python -m pip install --retries 10 numba")
        if cmd_return:
            logger.info("repo {} python -m pip install numba failed".format(self.reponame))
        cmd_return = os.system("python -m pip install --retries 10 sklearn")
        if cmd_return:
            logger.info("repo {} python -m pip install sklearn failed".format(self.reponame))
        cmd_return = os.system("python -m pip install --retries 10 pymatting")
        if cmd_return:
            logger.info("repo {} python -m pip install pymatting failed".format(self.reponame))
        #onnx
        cmd_return = os.system("python -m pip install --retries 10 paddle2onnx")
        if cmd_return:
            logger.info("repo {} python -m pip install paddle2onnx failed".format(self.reponame))
        cmd_return = os.system("python -m pip install --retries 10 onnxruntime==1.9.0")
        if cmd_return:
            logger.info("repo {} python -m pip install onnxruntime==1.9.0 failed".format(self.reponame))
        #cpp_infer
        os.chdir("test_tipc/cpp/")
        os.mkdir("inference_models")
        os.chdir("inference_models")
        if os.path.exists("pp_liteseg_infer_model"):
            logger.info("#### already download pp_liteseg_infer_model")
        else:
            logger.info("#### start download pp_liteseg_infer_model")
            self.download_data(
                value="https://paddleseg.bj.bcebos.com/dygraph/demo/pp_liteseg_infer_model.tar.gz"
            )
            logger.info("#### end download pp_liteseg_infer_model")
        os.chdir("../")
        if os.path.exists("paddle_inference"):
            logger.info("#### already download paddle_inference")
        else:
            logger.info("#### start download paddle_inference")
            self.download_data(
                value="https://paddle-qa.bj.bcebos.com/paddle-pipeline/Master_GpuAll_LinuxCentos_Gcc82_Cuda10.1_cudnn7.6_trt6015_onort_Py38_Compile_H/latest/paddle_inference.tgz"
            )
            os.rename("paddle_inference_install_dir", "paddle_inference")
            logger.info("#### end download paddle_inference")
        if os.path.exists("opencv-3.4.7"):
            logger.info("#### already download opencv-3.4.7")
        else:
            logger.info("#### start download opencv-3.4.7")
            self.download_data(
                value="https://paddleocr.bj.bcebos.com/libs/opencv/opencv-3.4.7.tar.gz"
            )
            logger.info("#### end download opencv-3.4.7")
        #os.chdir("opencv-3.4.7")
        #path_build = os.getcwd()
        #install_path = path_build + "/opencv3"
        #build_dir = path_build + "/build"
        #os.mkdir(build_dir)
        #os.chdir(build_dir)
        cmd = """
    cd opencv-3.4.7/
    root_path=$PWD
    install_path=${root_path}/opencv3
    build_dir=${root_path}/build

    rm -rf ${build_dir}
    mkdir ${build_dir}
    cd ${build_dir}

    cmake .. \
    -DCMAKE_INSTALL_PREFIX=${install_path} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \
    -DWITH_TIFF=ON \
    -DBUILD_TIFF=ON
    make -j
    make install
    cd ../../

    # build cpp
    bash build.sh
    """
        ###os.system(cmd) 

        #serving_python
        cmd_return = os.system("python -m pip install --retries 10 paddle-serving-server-gpu==0.9.0.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        if cmd_return:
            logger.info("repo {} python -m pip install paddle-serving-server-gpu==0.9.0.post102 failed".format(self.reponame))
        cmd_return = os.system("python -m pip install --retries 10 paddle_serving_client==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        if cmd_return:
            logger.info("repo {} python -m pip install paddle_serving_client==0.9.0 failed".format(self.reponame))
        cmd_return = os.system("python -m pip install --retries 10 paddle-serving-app==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        if cmd_return:
            logger.info("repo {} python -m pip install paddle-serving-app==0.9.0 failed".format(self.reponame))

        #serving_cpp
        cmd = """
        #apt-get update
        #apt install -y libcurl4-openssl-dev libbz2-dev
        yum install -y openssl openssl-devel
        yum install bzip2-devel
        wget -nv https://paddle-serving.bj.bcebos.com/others/centos_ssl.tar && tar xf centos_ssl.tar && rm -rf centos_ssl.tar && mv libcrypto.so.1.0.2k /usr/lib/libcrypto.so.1.0.2k && mv libssl.so.1.0.2k /usr/lib/libssl.so.1.0.2k && ln -sf /usr/lib/libcrypto.so.1.0.2k /usr/lib/libcrypto.so.10 && ln -sf /usr/lib/libssl.so.1.0.2k /usr/lib/libssl.so.10 && ln -sf /usr/lib/libcrypto.so.10 /usr/lib/libcrypto.so && ln -sf /usr/lib/libssl.so.10 /usr/lib/libssl.so

        # 安装go依赖
        rm -rf /usr/local/go
        wget -nv -qO- https://paddle-ci.cdn.bcebos.com/go1.17.2.linux-amd64.tar.gz | tar -xz -C /usr/local
        export GOROOT=/usr/local/go
        export GOPATH=/root/gopath
        export PATH=$PATH:$GOPATH/bin:$GOROOT/bin
        go env -w GO111MODULE=on
        go env -w GOPROXY=https://goproxy.cn,direct
        go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway@v1.15.2
        go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger@v1.15.2
        go install github.com/golang/protobuf/protoc-gen-go@v1.4.3
        go install google.golang.org/grpc@v1.33.0
        go env -w GO111MODULE=auto
        # 下载opencv库
        wget -nv https://paddle-qa.bj.bcebos.com/PaddleServing/opencv3.tar.gz && tar -xvf opencv3.tar.gz && rm -rf opencv3.tar.gz
        export OPENCV_DIR=$PWD/opencv3
        # clone Serving
        HTTP_PROXY=http://172.19.57.45:3128
        HTTPS_PROXY=http://172.19.57.45:3128
       
        set http_proxy=${HTTP_PROXY}
        set https_proxy=${HTTPS_PROXY}
        git clone https://github.com/PaddlePaddle/Serving.git -b v0.9.0 --depth=1
        cd Serving
        export Serving_repo_path=$PWD
        git submodule update --init --recursive
        set http_proxy=
        set https_proxy=

        python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        python -m pip install --retries 10 -r python/requirements.txt

        # set env
        export PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
        export PYTHON_LIBRARIES=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
        export PYTHON_EXECUTABLE=`which python`

        export CUDA_PATH='/usr/local/cuda'
        export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
        export CUDA_CUDART_LIBRARY='/usr/local/cuda/lib64/'
        export TENSORRT_LIBRARY_PATH='/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/'
        cd ..
        rm -f ${Serving_repo_path}/core/general-server/op/general_clas_op.*
        cp test_tipc/serving_cpp/general_seg_op.* ${Serving_repo_path}/core/general-server/op
        #build server
        cd Serving/
        rm -rf server-build-gpu-opencv
        mkdir server-build-gpu-opencv && cd server-build-gpu-opencv
        set http_proxy=${HTTP_PROXY}
        set https_proxy=${HTTPS_PROXY}
        cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
            -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
            -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
            -DCUDNN_LIBRARY=${CUDNN_LIBRARY} \
            -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY} \
            -DTENSORRT_ROOT=${TENSORRT_LIBRARY_PATH} \
            -DOPENCV_DIR=${OPENCV_DIR} \
            -DWITH_OPENCV=ON \
            -DSERVER=ON \
            -DWITH_GPU=ON ..
        make -j32

        # 安装serving ， 设置环境变量
        python -m pip install python/dist/paddle*
        export SERVING_BIN=$PWD/core/general-server/serving
        unset http_proxy
        unset https_proxy
        cd  ../../
        """

        ###os.system(cmd) 


        

        os.chdir(path_now)

        if 1 == 0:
            logger.info("#### slim install")
            if os.path.exists("PaddleSlim") is False:
                try:
                    wget.download("https://xly-devops.bj.bcebos.com/PaddleTest/PaddleSlim.tar.gz")
                    tf = tarfile.open("PaddleSlim.tar.gz")
                    tf.extractall(os.getcwd())
                except:
                    logger.info("#### prepare download failed {} failed".format("PaddleSlim.tar.gz"))
            if os.path.exists("PaddleSlim"):
                path_now = os.getcwd()
                os.chdir("PaddleSlim")
                os.system("git checkout develop")
                os.system("git pull")
                os.system("python -m pip install -r requirements.txt")
                cmd_return = os.system("python setup.py install")
                os.chdir(path_now)
            if cmd_return:
                logger.info("repo {} python -m pip install paddleslim failed".format(self.reponame))
                # return 1

        return 0


    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleSeg_Build, self).build_env()
        ret = 0
        ret = self.build_paddleseg()
        if ret:
            logger.info("build env whl failed")
            return ret
        #ret = self.build_yaml()
        #if ret:
        #    logger.info("build env yaml failed")
        #    return ret
        ret = self.build_dataset()
        if ret:
            logger.info("build env dataset failed")
            return ret
        return ret


if __name__ == "__main__":

    def parse_args():
        """
        接收和解析命令传入的参数
            最好尽可能减少输入给一些默认参数就能跑的示例!
        """
        parser = argparse.ArgumentParser("Tool for running CE task")
        parser.add_argument("--models_file", help="模型列表文件", type=str, default=None)
        parser.add_argument("--reponame", help="输入repo", type=str, default=None)
        args = parser.parse_args()
        return args

    args = parse_args()

    model = PaddleSeg_Build(args)
    model.build_paddleseg()
    model.build_dataset()

