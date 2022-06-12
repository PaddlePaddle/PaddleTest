filename=${1##*/}
#echo $filename
model=${filename%.*}
echo $model
log_path=log
cd ${Project_path}
echo '#####ls'
ls

# if [[ $2 == 'train_linux_gpu1' ]] ; then
#     echo '#####train_linux_gpu1'
#     cat ${log_path}/train/${model}_1card.log | grep Avg
#     cat ${log_path}/train/${model}_1card.log | grep Train | grep Avg | grep '5/5' > tmp_1card.log
#     cat ${log_path}/train/${model}_1card.log | grep Eval | grep Avg > tmp_1card1.log
#     sed -i 's/loss/train_eval/' tmp_1card1.log
#     cat tmp_1card1.log
#     cat tmp_1card1.log >> tmp_1card.log
#     cat ${log_path}/train/${model}_1card.log | grep exit_code
#     cat ${log_path}/train/${model}_1card.log | grep exit_code >> tmp_1card.log
#     cat tmp_1card.log | tr '\n' ',' > ../${log_path}/${model}_1card.log
#     cat ../${log_path}/${model}_1card.log

if [[ $2 == 'train_linux_gpu1' ]] ; then
    echo '#####train_linux_gpu1'
    tail -n 10 ${log_path}/train/${model}_1card.log

    cat ${log_path}/train/${model}_1card.log | grep Iter | grep '20/20'
    cat ${log_path}/train/${model}_1card.log | grep Iter | grep '20/20' >tmp_1card.log
    cat ${log_path}/train/${model}_1card.log | grep training_exit_code
    cat ${log_path}/train/${model}_1card.log | grep training_exit_code >>tmp_1card.log
    cat tmp_1card.log | tr '\n' ',' > ../${log_path}/${model}_1card.log
    cat ../${log_path}/${model}_1card.log

elif [[ $2 == 'train_linux_gpu2' ]] ; then
    echo '#####train_linux_gpu2'
    tail -n 10 ${log_path}/train/${model}_2card.log

    cat ${log_path}/train/${model}_2card.log | grep Iter | grep '20/20'
    cat ${log_path}/train/${model}_2card.log | grep Iter | grep '20/20' >tmp_2card.log
    cat ${log_path}/train/${model}_2card.log | grep training_exit_code
    cat ${log_path}/train/${model}_2card.log | grep training_exit_code >>tmp_2card.log
    cat tmp_2card.log | tr '\n' ',' > ../${log_path}/${model}_2card.log
    cat ../${log_path}/${model}_2card.log

elif [[ $2 == 'eval_linux' ]] ; then
    echo '#####eval_linux'
    tail -n 10 ${log_path}/eval/${model}.log

    cat ${log_path}/eval/${model}.log | grep eval_exit_code
    cat ${log_path}/eval/${model}.log | grep eval_exit_code >../${log_path}/${model}_eval.log
    cat ../${log_path}/${model}_eval.log

else
    echo '##### unknown type'
fi
