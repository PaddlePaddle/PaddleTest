cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型样例测试阶段"
#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/model_zoo/$model_name/
log_path=$root_path/log/$model_name/

#访问RD程序
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
python -u ./export_model.py \
    --input_model_dir ./SST-2/sst-2_ft_model_40.pdparams/ \
    --output_model_dir ./ \
    --model_name electra-deploy


python -u ./deploy/python/predict.py \
    --model_file ./electra-deploy.pdmodel \
    --params_file ./electra-deploy.pdiparams \
    --predict_sentences "uneasy mishmash of styles and genres ." "director rob marshall went out gunning to make a great one ." \
    --batch_size 2 \
    --max_seq_length 128 \
    --model_name electra-small > $log_path/infer.log 2>&1

print_info $? "infer"
