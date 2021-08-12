cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型样例测试阶段"
#路径配置
root_path=$cur_path/../../
log_path=$root_path/log/$model_name/
mkdir -p $log_path

print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $root_path/models_repo/examples/sentiment_analysis/skep

python predict_sentence.py \
    --model_name "skep_ernie_1.0_large_ch"\
    --device $1\
    --params_path checkpoints/model_300/model_state.pdparams > $log_path/infer_$1.log 2>&1

print_info $? infer_$1
