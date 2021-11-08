filename=${1##*/}
#echo $filename
model=${filename%.*}
echo $model
log_path=log
cd ${Project_path}
echo '#####ls'
ls
ls ${log_path}/
ls ${log_path}/train/
if [[ $2 == 'train_linux_gpu1' ]] ; then 
    echo '#####train_linux_gpu1'
    cat ${log_path}/train/${model}_1card.log | grep Train | grep Avg | grep 'Epoch 5/5'
    cat ${log_path}/train/${model}_1card.log | grep Train | grep Avg | grep 'Epoch 5/5' > ../log/${model}_1card.log

elif [[ $2 == 'train_linux_gpu2' ]] ; then 
    echo '#####train_linux_gpu2'
    cat ${log_path}/train/${model}_2card.log | grep Train | grep Avg | grep 'Epoch 5/5'
    cat ${log_path}/train/${model}_2card.log | grep Train | grep Avg | grep 'Epoch 5/5' > ../log/${model}_2card.log

elif [[ $2 == 'eval_linux' ]] ; then 
    echo '#####eval_linux'
    cat ${log_path}/train/${model}_2card.log | grep Avg
    cat ${log_path}/train/${model}_2card.log | grep Avg > ../log/${model}_eval.log

elif [[ $2 == 'infer_linux' ]] ; then 
    echo '#####infer_linux'
    cat ${log_path}/result.log | grep infer_exit_code
    cat ${log_path}/result.log | grep infer_exit_code >../log/${model}_infer.log

elif [[ $2 == 'export_linux' ]] ; then 
    echo '#####export_linux'
    cat ${log_path}/result.log | grep export_model_exit_code
    cat ${log_path}/result.log | grep export_model_exit_code >../log/${model}_export.log

elif [[ $2 == 'predict_linux' ]] ; then 
    echo '#####predict_linux'
    cat ${log_path}/result.log | grep predict_exit_code
    cat ${log_path}/result.log | grep predict_exit_code >../log/${model}_predict.log
fi