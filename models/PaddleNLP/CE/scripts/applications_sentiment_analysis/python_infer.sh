
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../applications/sentiment_analysis/
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
python  export_model.py \
    --model_type "extraction" \
    --model_path "./checkpoints/ext_checkpoints/best.pdparams" \
    --save_path "./checkpoints/ext_checkpoints/static/infer"
python  export_model.py \
    --model_type "classification" \
    --model_path "./checkpoints/cls_checkpoints/best.pdparams" \
    --save_path "./checkpoints/cls_checkpoints/static/infer"
cd deploy
python predict.py \
    --base_model_name "skep_ernie_1.0_large_ch" \
    --ext_model_path "../checkpoints/ext_checkpoints/static/infer" \
    --cls_model_path "../checkpoints/cls_checkpoints/static/infer" \
    --ext_label_path "../data/ext_data/label.dict" \
    --cls_label_path "../data/cls_data/label.dict" \
    --test_path "../data/test.txt" \
    --save_path "../data/sentiment_results.json" \
    --batch_size 8 \
    --ext_max_seq_len 512 \
    --cls_max_seq_len 256 > $log_path/python_infer_$1.log 2>&1

print_info $? python_infer_$1
