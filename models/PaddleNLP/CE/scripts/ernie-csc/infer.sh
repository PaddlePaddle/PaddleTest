#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_correction/ernie-csc/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path

python predict_sighan.py \
    --model_name_or_path ernie-1.0 \
    --test_file sighan_test/sighan13/input.txt \
    --batch_size 32 \
    --ckpt_path checkpoints/single/best_model.pdparams \
    --predict_file predict_sighan13.txt > $log_path/infer_$1.log 2>&1
print_info $? infer_$1
