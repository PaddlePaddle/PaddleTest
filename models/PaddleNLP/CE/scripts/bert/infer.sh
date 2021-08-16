cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型样例测试阶段"
#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#访问RD程序
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
echo "当前CUDA配置"
echo $CUDA_VISIBLE_DEVICES
cd $code_path

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

python -u ./export_model.py \
    --model_type bert \
    --model_path bert-base-uncased \
    --output_path ./infer_model/model

python -u ./predict_glue.py \
    --task_name SST-2 \
    --model_type bert \
    --model_path ./infer_model/model \
    --batch_size 32 \
    --max_seq_length 128 > $log_path/bert_predict.log 2>&1

print_info $? "bert_predict"
#cat $log_path/bert_predict.log
