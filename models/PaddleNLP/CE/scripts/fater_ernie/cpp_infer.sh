#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/experimental/faster_ernie/$2/
log_path=$root_path/log/$model_name/
DPADDLE_LIB=$cur_path/../../models_repo/examples/experimental/faster_ernie/$2/cpp_deploy/lib/paddle_inference

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path

mkdir -p cpp_deploy/lib
cd cpp_deploy/lib
if nvcc -V | grep 10.2; then
    wget https://paddle-qa.bj.bcebos.com/paddle-pipeline/Master_Gpu_All_Linux_Gcc82_Cuda10.2_cudnn7.6_trt6018_Py38_Compile_H/latest/paddle_inference.tgz
else 
    wget https://paddle-qa.bj.bcebos.com/paddle-pipeline/Master_Gpu_All_Linux_Gcc82_Cuda10.2_cudnn7.6_trt6018_Py38_Compile_H/latest/paddle_inference.tgz
fi
tar -xzvf paddle_inference.tgz
mv paddle_inference_install_dir paddle_inference
cd ..

mkdir build
cd build
cmake .. -DPADDLE_LIB=$DPADDLE_LIB \
    -DWITH_MKL=ON \
    -DPROJECT_NAME=$2_infer \
    -DWITH_GPU=ON \
    -DWITH_STATIC_LIB=OFF
make -j
./$2_infer --model_file ../../export/inference.pdmodel --params_file ../../export/inference.pdiparams > cpp_infer_$2_$1.log 2>&1 
print_info $? cpp_infer_$2_$1