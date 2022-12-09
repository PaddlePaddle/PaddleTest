export model_dir=$1
export example=$2
export device=gpu
export epoch=1
export max_steps=2
export save_steps=2
export output_dir=./output/
export export_dir=./infer_model/

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
    if [[ ${exec_file} == "requirement" ]];then
        python -m pip install -r ${exec_file}
    elif [[ ${exec_file} == "run_prepare.py" ]];then
        python run_prepare.py  >${log_path}/${example}_prepare >>${log_path}/${example}_prepare 2>&1
        print_info $? ${example}_prepare
    elif  [[ ${exec_file} == "run_prepare.sh" ]];then
        bash run_prepare.sh  >${log_path}/${example}_prepare >>${log_path}/${example}_prepare 2>&1
        print_info $? ${example}_prepare
    # TRAIN
    elif [[ ${exec_file} == "train.py" ]] || [[ ${exec_file} =~ "run_train" ]] ;then
        python -m paddle.distributed.launch ${exec_file} --device ${devices} --max_steps ${max_steps} --save_steps ${save_steps} --output_dir ${output_dir} >${log_path}/${example}_train>>${log_path}/${example}_train 2>&1
        print_info $? ${example}_train
    # EVAL
    elif [[ ${exec_file} == "eval.py" ]] || [[ ${exec_file} =~ "run_eval" ]] ;then
        python ${exec_file} --device ${devices}  --init_checkpoint_dir $output_dir}  >${log_path}/${example}_eval >>${log_path}/${example}_eval 2>&1
        print_info $? ${example}_eval
    # PREDICT
    elif [[ ${exec_file} == "predict.py" ]] || [[ ${exec_file} =~ "run_predict" ]] ;then
        python ${exec_file} --device ${devices} --init_checkpoint_dir ${output_dir}   >${log_path}/${example}_predict >>${log_path}/${example}_predcit 2>&1
        print_info $? ${example}_predict
    # EXPORT MODEL
    elif [[ ${exec_file} == "export_model.py" ]] ;then
        python ${exec_file} --device ${devices} --export_dir ${export_dir} >${log_path}/${example}_export_model >>${log_path}/${example}_export_model 2>&1
        print_info $? ${example}_export_model
    # INFER
    elif [[ ${exec_file} == "infer.py" ]] || [[ ${exec_file} =~ "run_infer" ]] ;then
        python ${exec_file} --device ${devices} --infer_dir ${export_dir} >${log_path}/${example}_predict >>${log_path}/${example}_predcit 2>&1
        print_info $? ${example}_predict
    # CUSTOM
    elif [[ ${exec_file} =~ "run_" ]] && [[ ${exec_file##*.} == "py" ]];then
        python ${exec_file} >${log_path}/${example}_${exec_file%%.*} >>${log_path}/${example}_${exec_file%%.*} 2>&1
        print_info $? ${example}_${exec_file%%.*}
    elif [[ ${exec_file} =~ "run_" ]] && [[ ${exec_file##*.} == "sh" ]];then
        bash ${exec_file}  >${log_path}/${example}_${exec_file%%.*} >>${log_path}/${example}_${exec_file%%.*} 2>&1
        print_info $? ${example}_${exec_file%%.*}
    fi
done
}
run_example
