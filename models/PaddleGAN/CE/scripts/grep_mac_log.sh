filename=${1##*/}
#echo $filename
model=${filename%.*}
echo $model
log_path=log
cd ${Project_path}
echo '#####ls'
ls

if [[ $2 == 'train_mac_cpu' ]] ; then
    echo '#####train_mac_cpu'
    tail -n 5 ${log_path}/train/${model}.log

    cat ${log_path}/train/${model}_cpu.log | grep training_exit_code
    cat ${log_path}/train/${model}_cpu.log | grep training_exit_code >../${log_path}/${model}_cpu.log
    cat ../${log_path}/${model}_cpu.log

elif [[ $2 == 'eval_mac' ]] ; then
    echo '#####eval_mac'
    tail -n 5 ${log_path}/eval/${model}.log

    cat ${log_path}/eval/${model}.log | grep eval_exit_code
    cat ${log_path}/eval/${model}.log | grep eval_exit_code >../${log_path}/${model}_eval.log
    cat ../${log_path}/${model}_eval.log

else
    echo '##### unknown type'
fi
