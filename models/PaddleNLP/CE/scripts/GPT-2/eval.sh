cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型样例评估阶段"

#路径配置
root_path=$cur_path/../../
log_path=$root_path/log/$model_name/
mkdir -p $log_path

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $root_path/models_repo
cd examples/language_model/gpt

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip

python run_eval.py --model_name gpt2-en \
    --eval_path ./wikitext-103/wiki.valid.tokens \
    --overlapping_eval 32 \
    --init_checkpoint_path ./output/model_10/model_state.pdparams \
    --batch_size 2 \
    --device $1 > $log_path/eval_$1.log 2>&1

print_info $? eval_$1
