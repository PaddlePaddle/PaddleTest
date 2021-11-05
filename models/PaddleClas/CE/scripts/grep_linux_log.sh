filename=${1##*/}
#echo $filename
model=${filename%.*}
echo $model
log_path=log
cd ${Project_path}

if [[ $2 == 'train_linux_gpu1' ]] ; then 
    cat ${log_path}/train/${model}_1card.log | grep Train | grep Avg | grep 'Epoch 5/5' > ../log/${model}_1card.log

elif [[ $2 == 'train_linux_gpu2' ]] ; then 
    cat ${log_path}/train/${model}_2card.log | grep Train | grep Avg | grep 'Epoch 5/5' > ../log/${model}_2card.log

elif [[ $2 == 'eval_linux' ]] ; then 
    cat ${log_path}/train/${model}_2card.log | grep Avg > ../log/${model}_eval.log

elif [[ $2 == 'infer_linux' ]] ; then 
    cat ${log_path}/result.log | grep infer_exit_code >../log/${model}_infer.log

elif [[ $2 == 'export_linux' ]] ; then 
    cat ${log_path}/result.log | grep export_model_exit_code >../log/${model}_export.log

elif [[ $2 == 'predict_linux' ]] ; then 
    cat ${log_path}/result.log | grep predict_exit_code >../log/${model}_predict.log



