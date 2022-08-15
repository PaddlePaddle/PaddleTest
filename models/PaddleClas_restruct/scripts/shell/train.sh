
# 输入变量：yaml、设置卡数 CPU SET_CUDA SET_MULTI_CUDA 、训练的模型动态图/静态图/收敛性( dynamic static convergence )

export yaml_line=${1:-ppcls/configs/ImageNet/ResNet/ResNet50.yaml}
export cuda_type=${2:-SET_MULTI_CUDA}
export train_type=${3:-dynamic}
export Project_path=${Project_path:-$PWD}

cd ${Project_path} #确定下执行路径
# ls
# ls ${Project_path}/../  #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
# ls ${Project_path}/../scripts
\cp -r -f ${Project_path}/../scripts/shell/prepare.sh .
source prepare.sh

#区分动态图、静态图
echo ${train_type}
if [[ ${train_type} =~ "dynamic" ]] || [[ ${train_type} =~ "convergence" ]];then
    export train_mold="tools/train.py"
else
    export train_mold="ppcls/static/train.py"
fi
echo ${train_mold}

sleep 3
#防止训练过程太快导致显存未释放hang
case ${train_type} in #动态图/静态图/收敛性
dynamic|static)
    echo "######  params_dir"
    echo ${params_dir}
    if [[ `cat ${yaml_line} |grep MultiScaleSampler|wc -l` -gt "0" ]]; then #for tingquan 220513 change
        echo "have MultiScaleSampler"
        common_par="-o Global.epochs=2 \
        -o Global.save_interval=2 \
        -o Global.eval_interval=2 \
        -o Global.seed=1234 \
        -o DataLoader.Train.loader.num_workers=0 \
        -o DataLoader.Train.sampler.first_bs=32 \
        -o Global.output_dir=${output_dir}/${model_name} \
        -o Global.device=${set_cuda_device}"
    else
        common_par="-o Global.epochs=2 \
        -o Global.save_interval=2 \
        -o Global.eval_interval=2 \
        -o Global.seed=1234 \
        -o DataLoader.Train.loader.num_workers=0 \
        -o DataLoader.Train.sampler.shuffle=False  \
        -o Global.output_dir=${output_dir}/${model_name} \
        -o Global.device=${set_cuda_device}"
    fi
    if [[ ${yaml_line} =~ 'GeneralRecognition' ]]; then
        python ${multi_flag} ${train_mold} -c ${yaml_line} \
            -o DataLoader.Train.sampler.batch_size=32 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    elif [[ ${yaml_line} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
        python ${multi_flag} ${train_mold} -c ${yaml_line} \
            -o DataLoader.Eval.sampler.batch_size=64 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    elif [[ ${yaml_line} =~ 'quantization' ]] ; then
        python ${multi_flag} ${train_mold} -c ${yaml_line} \
            -o DataLoader.Train.sampler.batch_size=32 \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    else
        python ${multi_flag} ${train_mold} -c ${yaml_line}  \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    fi
    # if ([[ -f "${output_dir}/${model_name}/${params_dir}/latest.pdparams" ]] \
    #     || [[ -f "${output_dir}/${model_name}/${params_dir}/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
    #     ;then
    echo " ${output_dir}/${model_name}/${params_dir} "
    echo $(ls ${output_dir}/${model_name}/${params_dir}|head -n 2)
    if [[ -f "${output_dir}/${model_name}/${params_dir}/latest.pdparams" ]] \
        || [[ -f "${output_dir}/${model_name}/${params_dir}/0/ppcls.pdmodel" ]];then
        # && [[ $(grep -c  "Error" ${log_path}/train/${model_name}_${card}.log) -eq 0 ]];then
        echo -e "\033[33m ${train_type} successfully! training in ${card} of ${model_name} successfully!\033[0m" \
            |tee -a ${log_path}/result.log
        echo "training_exit_code: 0.0" >> ${log_path}/train/${model_name}_${card}.log
    else
        cat ${log_path}/train/${model_name}_${card}.log
        echo -e "\033[31m ${train_type} failed! training in ${card} of ${model_name} failed!\033[0m" \
            |tee -a ${log_path}/result.log
        echo "training_exit_code: 1.0" >> ${log_path}/train/${model_name}_${card}.log
    fi
    # cat ${log_path}/train/${model_name}_${card}.log | grep "Memory Usage (MB)" #查看显存
;;
convergence)
    echo "######  params_dir"
    echo ${params_dir}
    python ${multi_flag} ${train_mold} -c ${yaml_line}  \
        -o Global.output_dir=${output_dir}/${model_name} \
        > ${log_path}/train/${model_name}_convergence.log 2>&1
    # if ([[ -f "${output_dir}/${model_name}/${params_dir}/latest.pdparams" ]] \
    #     || [[ -f "${output_dir}/${model_name}/${params_dir}/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
    #     ;then
    if [[ -f "${output_dir}/${model_name}/${params_dir}/latest.pdparams" ]] \
        || [[ -f "${output_dir}/${model_name}/${params_dir}/0/ppcls.pdmodel" ]];then
        # && [[ $(grep -c  "Error" ${log_path}/train/${model_name}_convergence.log) -eq 0 ]];then
        echo -e "\033[33m ${train_type} successfully! training in ${card} of ${model_name}  successfully!\033[0m" \
            |tee -a ${log_path}/result.log
        echo "training_multi_exit_code: 0.0" >> ${log_path}/train/${model_name}_convergence.log
    else
        cat ${log_path}/train/${model_name}_convergence.log
        echo -e "\033[31m ${train_type} failed! training in ${card} of ${model_name} failed!\033[0m" \
            |tee -a ${log_path}/result.log
        echo "training_multi_exit_code: 1.0" >> ${log_path}/train/${model_name}_convergence.log
    fi
    # cat ${log_path}/train/${model_name}_convergence.log | grep "Memory Usage (MB)" #查看显存
;;
esac
