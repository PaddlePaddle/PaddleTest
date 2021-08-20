#!/usr/bin/env bash
cur_path=`pwd`
echo "++++++++++++++++++++++++++++++++$1 finetune!!!!!!!!!++++++++++++++++++++++++++++++++"
root_path=$cur_path/../../
log_path=$root_path/log/$1/
mkdir -p $log_path
print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/EXIT_$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/EXIT_$2.log
fi
}
python nlp_finetune.py --model_name $1 \
                       --use_gpu $2 \
                       --max_steps $3 \
                       --batch_size $4 \
                       --module_name $5 \
                       --author $6 > ${log_path}/${1}_finetune_$2_$3_$4_$5_$6.log 2>&1
print_info $? ${1}_finetune_$2_$3_$4_$5_$6
