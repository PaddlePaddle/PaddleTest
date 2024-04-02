
@ echo off

@REM #定义环境变量
setlocal enabledelayedexpansion
set FLAGS_cudnn_deterministic=True #固定随机量使用，使cuda算法保持一致

echo "######  ----ln  data-----"
cd dataset
if not exist ILSVRC2012 (mklink /j ILSVRC2012 %data_path%\PaddleClas\ILSVRC2012)
cd ..

@REM # paddle
echo "######  paddle version"
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

@REM # repo
echo "######  clas version"
git rev-parse HEAD

@REM # python
python -c 'import sys; print(sys.version_info[:])'
echo "######  python version"

@REM # system
echo "######  system windows"

@REM #取消代理用镜像安装包
set http_proxy=
set https_proxy=

python -m pip install --ignore-installed --upgrade  pip -i https://mirror.baidu.com/pypi/simple
python -m pip install  --ignore-installed paddleslim -i https://mirror.baidu.com/pypi/simple
python -m pip install  -r requirements.txt    -i https://mirror.baidu.com/pypi/simple


@REM #确定log存储位置
set log_path="../log"
set output_dir="output"

@REM #区分GPU和CPU
set set_cuda_device=%2
if !cpu_gpu! == GPU (
    set set_cuda_device=gpu
    set set_cuda_flag=True
) else if !cpu_gpu! == CPU (
    set set_cuda_device=cpu
    set set_cuda_flag=False
)

@REM # 获取模型名称
set targe=%1
@REM set target=ppcls/configs/ImageNet/ResNet/ResNet50.yaml
set target1=%target:*/=%
set target2=%target1:*/=%
set target3=%target2:*/=%
set target4=%target3:*/=%
set target5=%target4:*/=%
set model=%target2:.yaml=%
set model_name=%model:/=_%
set model_latest=%target5:.yaml=%
set model_type=!target2:%target3%=!
set model_type=!model_type:/=!
echo %model_name%
echo %model_latest%
echo %model_type%
