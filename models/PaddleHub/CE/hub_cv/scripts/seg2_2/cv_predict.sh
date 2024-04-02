#!/usr/bin/env bash
cur_path=`pwd`
echo "++++++++++++++++++++++++++++++++$1 predict!!!!!!!!!++++++++++++++++++++++++++++++++"
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
python cv_predict.py --model_name $1 \
                     --use_gpu $2 \
                     --visualization $3 \
                     --batch_size $4 \
                     --input_size $5 \
                     --output_dir $6 \
                     --img_path $7 > ${log_path}/${1}_predict_$2_$3_$4_$5.log 2>&1
print_info $? ${1}_predict_$2_$3_$4_$5
