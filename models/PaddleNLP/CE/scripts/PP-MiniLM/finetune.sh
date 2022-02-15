cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/pp-minilm/
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

TASK=$3
LR=$4
BS=$5
MAX_SEQ_LEN=$6

#访问RD程序
cd $code_path/finetuning
mkdir ppminilm-6l-768h
sh run_clue.sh ${TASK} ${LR} ${BS} 1 ${MAX_SEQ_LEN} $2 ppminilm-6l-768h > ${log_path}/finetune_${TASK}_${LR}_${BS}_$1.log
print_info $? finetune_${TASK}_${LR}_${BS}_$1
#导出模型，失败则覆盖原来的退出码
python export_model.py --task_name ${TASK} --model_path ppminilm-6l-768h/models/${TASK}/${LR}_${BS}
print_info $? finetune_${TASK}_${LR}_${BS}_$1
