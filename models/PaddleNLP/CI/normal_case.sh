export model_dir=$1
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/$2_FAIL.log
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
    cat ${log_path}/$2_FAIL.log
else
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
run_example(){
cd $model_dir
for exec_file in `ls`;do
    # Data Prepare
    if [[ ${exec_file} == "run_prepare.py" ]];then
        python run_prepare.py  >${log_path}/${example}_prepare >>${log_path}/${example}_prepare 2>&1
        print_info $? ${example}_prepare
    elif  [[ ${exec_file} == "run_prepare.sh" ]];then
        bash run_prepare.sh  >${log_path}/${example}_prepare >>${log_path}/${example}_prepare 2>&1
        print_info $? ${example}_prepare
    # TRAIN
    elif [[ ${exec_file} == "train.py" ]] || [[ ${exec_file} =~ "run_train" ]] ;then
        python -m paddle.distributed.launch ${exec_file} --devices ${devices} --epoch ${epoch} --max_steps ${max_steps} --save_steps ${save_steps} --output ${output}  >${log_path}/${example}_train>>${log_path}/${example}_train 2>&1
        print_info $? ${example}_train
    # EVAL
    elif [[ ${exec_file} == "eval.py" ]] || [[ ${exec_file} =~ "run_eval" ]] ;then
        python ${exec_file} --devices ${devices}  --checkpoint ${output}  >${log_path}/${example}_eval >>${log_path}/${example}_eval 2>&1
        print_info $? ${example}_eval
    # PREDICT
    elif [[ ${exec_file} == "predict.py" ]] || [[ ${exec_file} =~ "run_predict" ]] ;then
        python ${exec_file} --devices ${devices}  --checkpoint ${output}  >${log_path}/${example}_predict >>${log_path}/${example}_predcit 2>&1
        print_info $? ${example}_predict
    # EXPORT_MODEL
    elif [[ ${exec_file} == "export_model.py" ]] ;then
        python ${exec_file} --devices ${devices}  --checkpoint ${output} --infer_dir ${infer_dir}  >${log_path}/${example}_export_model >>${log_path}/${example}_export_model 2>&1
        print_info $? ${example}_export_model
    # INFER
    elif [[ ${exec_file} == "infer.py" ]] || [[ ${exec_file} =~ "run_infer" ]] ;then
        python ${exec_file} --devices ${devices}  --infer_dir ${infer_dir}   >${log_path}/${example}_predict >>${log_path}/${example}_predcit 2>&1
        print_info $? ${example}_predict
    # CUSTOM
    elif [[ ${exec_file} =~ "run_" ]] && [[ ${exec_file##*.} == "py" ]];then
        python ${exec_file}  >${log_path}/${example}_${exec_file} >>${log_path}/${example}_${exec_file} 2>&1
        print_info $? ${example}_${exec_file}
    elif [[ ${exec_file} =~ "run_" ]] && [[ ${exec_file##*.} == "sh" ]];then
        sh ${exec_file}  >${log_path}/${example}_${exec_file} >>${log_path}/${example}_${exec_file} 2>&1
        print_info $? ${example}_${exec_file}
    fi
done
}
run_example
