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
    cat ${log_path}/train/${model}.log | grep training_exit_code
    cat ${log_path}/train/${model}.log | grep training_exit_code >../${log_path}/${model}_cpu.log
    cat ../${log_path}/${model}_cpu.log

elif [[ $2 == 'eval_mac' ]] ; then
    echo '#####eval_mac'
    tail -n 5 ${log_path}/eval/${model}.log
    cat ${log_path}/eval/${model}.log | grep eval_exit_code
    cat ${log_path}/eval/${model}.log | grep eval_exit_code >../${log_path}/${model}_eval.log
    cat ../${log_path}/${model}_eval.log

elif [[ $2 == 'infer_mac' ]] ; then
    echo '#####infer_mac'
    tail -n 5 ${log_path}/infer/${model}.log
    cat ${log_path}/infer/${model}.log | grep infer_exit_code
    cat ${log_path}/infer/${model}.log | grep infer_exit_code >../${log_path}/${model}_infer.log
    cat ../${log_path}/${model}_infer.log

elif [[ $2 == 'export_mac' ]] ; then
    echo '#####export_mac'
    cat ${log_path}/export/${model}.log | grep export_exit_code
    cat ${log_path}/export/${model}.log | grep export_exit_code >../${log_path}/${model}_export.log
    cat ../${log_path}/${model}_export.log

elif [[ $2 == 'predict_mac' ]] ; then
    echo '#####predict_mac'
    tail -n 5 ${log_path}/predict/${model}.log
    cat ${log_path}/predict/${model}.log | grep predict_exit_code
    cat ${log_path}/predict/${model}.log | grep predict_exit_code >../${log_path}/${model}_predict.log
    cat ../${log_path}/${model}_predict.log

else
    echo '##### unknown type'

fi
