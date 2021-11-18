filename=${1##*/}
#echo $filename
model=${filename%.*}
echo $model
log_path=log
cd ${Project_path}
echo '#####ls'
ls

if [[ $2 == 'train_linux_gpu1' ]] ; then
    echo '#####train_linux_gpu1'
    cat ${log_path}/train/${model}_1card.log  | grep Avg 
    cat ${log_path}/train/${model}_1card.log | grep Train | grep Avg | grep 'Epoch 5/5' > tmp_1card.log
    cat ${log_path}/train/${model}_1card.log | grep Eval | grep Avg >> tmp_1card.log
    sed -i '2s/loss/train_eval/' tmp_1card.log
    cat tmp_1card.log | tr '\n' ' ' > ../${log_path}/${model}_1card.log

elif [[ $2 == 'train_linux_gpu2' ]] ; then
    echo '#####train_linux_gpu2'
    cat ${log_path}/train/${model}_2card.log | grep Avg 
    cat ${log_path}/train/${model}_2card.log | grep Train | grep Avg | grep 'Epoch 5/5' > tmp_2card.log
    cat ${log_path}/train/${model}_2card.log | grep Eval | grep Avg >> tmp_2card.log
    sed -i '2s/loss/train_eval/' tmp_2card.log
    cat tmp_2card.log | tr '\n' ' ' > ../${log_path}/${model}_2card.log

elif [[ $2 == 'eval_linux' ]] ; then
    echo '#####eval_linux'
    cat ${log_path}/eval/${model}.log | grep Avg
    cat ${log_path}/eval/${model}.log | grep Eval | grep Avg > ../${log_path}/${model}_eval.log

elif [[ $2 == 'infer_linux' ]] ; then
    echo '#####infer_linux'
    cat ${log_path}/result.log | grep infer_exit_code
    cat ${log_path}/result.log | grep infer_exit_code >../${log_path}/${model}_infer.log

elif [[ $2 == 'export_linux' ]] ; then
    echo '#####export_linux'
    cat ${log_path}/result.log | grep export_exit_code
    cat ${log_path}/result.log | grep export_exit_code >../${log_path}/${model}_export.log

elif [[ $2 == 'predict_linux' ]] ; then
    echo '#####predict_linux'
    cat ${log_path}/result.log | grep predict_exit_code
    cat ${log_path}/result.log | grep predict_exit_code >../${log_path}/${model}_predict.log
fi
