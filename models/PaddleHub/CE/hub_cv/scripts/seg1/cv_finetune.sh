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
python cv_finetune.py --model_name $1 \
                       --use_gpu $2 \
                       --batch_size $3 \
                       --num_epoch $4 \
                       --optimizer $5 \
                       --learning_rate $6 \
                       --target_size $7 \
                       --class_num $8 \
                       --save_interval $9 \
                       --checkpoint_dir ${10} > ${log_path}/${1}_finetune_$2_$3_$4_$5_$6_$7_$8_$9_${10}.log 2>&1
print_info $? ${1}_finetune_$2_$3_$4_$5_$6_$7_$8_$9_${10}
