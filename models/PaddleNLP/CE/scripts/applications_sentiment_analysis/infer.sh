
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/applications/sentiment_analysis/
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

cp -r $cur_path/test.txt  $code_path/data/test.txt

#访问RD程序
cd $code_path
python predict.py \
    --ext_model_path "./checkpoints/ext_checkpoints/best.pdparams" \
    --cls_model_path "./checkpoints/cls_checkpoints/best.pdparams" \
    --test_path "./data/test.txt" \
    --ext_label_path "./data/ext_data/label.dict" \
    --cls_label_path "./data/cls_data/label.dict" \
    --save_path "./data/sentiment_results.json" \
    --batch_size 8 \
    --ext_max_seq_len 512 \
    --cls_max_seq_len 256 > $log_path/infer_$1.log 2>&1

print_info $? infer_$1
