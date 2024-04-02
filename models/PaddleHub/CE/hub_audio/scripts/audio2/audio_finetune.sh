#!/usr/bin/env bash
cur_path=`pwd`
echo "++++++++++++++++++++++++++++++++$1 begin to finetune !!!!!!!!!++++++++++++++++++++++++++++++++"
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
python audio_finetune.py --model_name $1 \
                         --task $2 \
                         --use_gpu $3 \
                         --batch_size $4 \
                         --num_epoch $5 \
                         --learning_rate $6 \
                         --save_interval $7 \
                         --checkpoint_dir $8 > ${log_path}/${1}_finetune_$2_$3_$4_$5_$6_$7.log 2>&1
print_info $? ${1}_finetune_$2_$3_$4_$5_$6_$7
