cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型Fine-tune阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
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

cd $code_path

sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
sed -i "s#init_from_params: .*#init_from_params: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
python ./eval.py --config ./configs/enwik8.yaml > $log_path/enwiki_eval.log 2>&1
print_info $? "enwiki_eval"
