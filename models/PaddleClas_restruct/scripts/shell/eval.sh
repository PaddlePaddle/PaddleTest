# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA 、 trained/pretrained

export yaml_line=${1:-ppcls/configs/ImageNet/ResNet/ResNet50.yaml}
export cuda_type=${2:-SET_MULTI_CUDA}
export eval_type=${3:-pretrained}

cd ${Project_path} #确定下执行路径
\cp -r -f ${Project_path}/../scripts/shell/prepare.sh . # #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
\cp -r -f ${Project_path}/../scripts/shell/choose_model.sh .

# 廷权临时增加规则 220413
if [[ ${yaml_line} =~ 'ultra' ]];then
    cp ${yaml_line} ${yaml_line}_tmp
    sed -i '/output_fp16: True/d' ${yaml_line}
fi

source prepare.sh
# arr=("trained" "pretrained") #或者抽象出来到输入参数，现在是默认训好的、预训练的全跑
# for eval_type in ${arr[@]}
# do
source choose_model.sh

if [[ ${yaml_line} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
    python ${multi_flag} tools/eval.py -c ${yaml_line} \
        -o Global.pretrained_model=${pretrained_model} \
        -o DataLoader.Eval.sampler.batch_size=64 \
        -o Global.output_dir=${output_dir}/${model_name} \
        > ${log_path}/eval/${model_name}_${eval_type}.log 2>&1
else
    python ${multi_flag} tools/eval.py -c ${yaml_line} \
        -o Global.pretrained_model=${pretrained_model} \
        -o Global.output_dir=${output_dir}/${model_name} \
        > ${log_path}/eval/${model_name}_${eval_type}.log 2>&1
fi

# if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ${log_path}/eval/${model_name}_${eval_type}.log) -eq 0 ]];then
if [[ $? -eq 0 ]];then
    echo -e "\033[33m eval of ${model_name}_${eval_type}  successfully!\033[0m"| tee -a ${log_path}/result.log
    echo "eval_exit_code: 0.0" >> ${log_path}/eval/${model_name}_${eval_type}.log
else
    cat ${log_path}/eval/${model_name}_${eval_type}.log
    echo -e "\033[31m eval of ${model_name}_${eval_type} failed!\033[0m" | tee -a ${log_path}/result.log
    echo "eval_exit_code: 1.0" >> ${log_path}/eval/${model_name}_${eval_type}.log
fi

# done

# 廷权临时增加规则 220413
if [[ ${yaml_line} =~ 'ultra' ]];then
    rm -rf ${yaml_line}
    mv ${yaml_line}_tmp ${yaml_line}
fi
