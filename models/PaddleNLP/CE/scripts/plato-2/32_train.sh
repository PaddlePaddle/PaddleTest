#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/dialogue/plato-2/
log_path=$root_path/log/$model_name/

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

#访问RD程序
cd $code_path

wget https://paddlenlp.bj.bcebos.com/models/transformers/plato2/32L.pdparams

if [[ $1 == "gpu" ]]; then
     python interaction.py\
          --vocab_path ./data/vocab.txt\
          --spm_model_file ./data/spm.model\
          --num_layers 32\
          --init_from_ckpt ./32L.pdparams < input.txt  > $log_path/train_32_$2_$1.log 2>&1
     print_info $? train_32_$2_$1
fi
