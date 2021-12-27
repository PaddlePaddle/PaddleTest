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
    # cat ${log_path}/train/${model}_cpu.log | grep Avg
    # cat ${log_path}/train/${model}_cpu.log | grep Train | grep Avg | grep '2/2' > tmp_cpu.log
    # cat ${log_path}/train/${model}_cpu.log | grep Eval | grep Avg > tmp_cpu1.log
    # sed -i '' 's/loss/train_eval/g' tmp_cpu1.log
    # cat tmp_cpu1.log
    # cat tmp_cpu1.log >> tmp_cpu.log
    # cat ${log_path}/train/${model}_cpu.log | grep exit_code
    # cat ${log_path}/train/${model}_cpu.log | grep exit_code >> tmp_cpu.log
    # cat tmp_cpu.log | tr '\n' ',' > ../${log_path}/${model}_cpu.log
    # cat ../${log_path}/${model}_cpu.log

    cat ${log_path}/train/${model}_cpu.log | grep training_exit_code
    cat ${log_path}/train/${model}_cpu.log | grep training_exit_code >../${log_path}/${model}_cpu.log
    cat ../${log_path}/${model}_cpu.log

elif [[ $2 == 'eval_mac' ]] ; then
    echo '#####eval_mac'
    # cat ${log_path}/eval/${model}.log | grep Avg
    # cat ${log_path}/eval/${model}.log | grep Eval | grep Avg > tmp_eval.log
    # cat ${log_path}/eval/${model}.log | grep exit_code
    # cat ${log_path}/eval/${model}.log | grep exit_code  >> tmp_eval.log
    # cat tmp_eval.log | tr '\n' ',' > ../${log_path}/${model}_eval.log
    # cat ../${log_path}/${model}_eval.log

    cat ${log_path}/eval/${model}.log | grep eval_exit_code
    cat ${log_path}/eval/${model}.log | grep eval_exit_code >../${log_path}/${model}_eval.log
    cat ../${log_path}/${model}_eval.log

elif [[ $2 == 'infer_mac' ]] ; then
    echo '#####infer_mac'
    cat ${log_path}/infer/${model}.log | grep infer_exit_code
    cat ${log_path}/infer/${model}.log | grep infer_exit_code >../${log_path}/${model}_infer.log
    cat ../${log_path}/${model}_infer.log

elif [[ $2 == 'export_mac' ]] ; then
    echo '#####export_mac'
    cat ${log_path}/export_model/${model}.log | grep export_exit_code
    cat ${log_path}/export_model/${model}.log | grep export_exit_code >../${log_path}/${model}_export.log
    cat ../${log_path}/${model}_export.log

elif [[ $2 == 'predict_mac' ]] ; then
    echo '#####predict_mac'
    cat ${log_path}/predict/${model}.log | grep predict_exit_code
    cat ${log_path}/predict/${model}.log | grep predict_exit_code >../${log_path}/${model}_predict.log
    cat ../${log_path}/${model}_predict.log
fi
