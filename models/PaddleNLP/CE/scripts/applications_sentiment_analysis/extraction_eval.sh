
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/applications/sentiment_analysis/extraction/
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

python evaluate.py \
    --model_path "../checkpoints/ext_checkpoints/best.pdparams" \
    --test_path "../data/ext_data/test.txt" \
    --label_path "../data/ext_data/label.dict" \
    --batch_size 16 \
    --max_seq_len 256 > $log_path/extraction_eval_$1.log 2>&1

print_info $? extraction_eval_$1
