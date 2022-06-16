#!/usr/bin/env bash
# 这里面的可以直接写训练case也可以跳转到script下（先将有差异的统一）
print_info(){
cat ${log_path}/$2.log
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
    #copy 一份失败的日志
    cp ${log_path}/$2.log ${log_path}/$2_FAIL.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}
# 1 waybill_ie
waybill_ie(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/waybill_ie/
    default_list="train,dy_to_st_infer"
else
    log_path=$log_path
    default_list="train"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    # 定义train阶段
    cd ./waybill_ie
    bash data_proc.sh
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} ernie > $log_path/train_${device}_ernie.log 2>&1
    print_info $? train_${device}_ernie
    bash train.sh ${device} bigru_crf > $log_path/train_${device}_bigru_crf.log 2>&1
    print_info $? train_${device}_bigru_crf
    bash train.sh ${device} ernie_crf > $log_path/train_${device}_ernie_crf.log 2>&1
    print_info $? train_${device}_ernie_crf
    cd ..
}
dy_to_st_infer(){
    cd ./waybill_ie
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash python_infer.sh ${device} ernie > ${log_path}/python_infer_ernie_${device}.log 2>&1
    print_info $? python_infer_ernie_${device}
    bash python_infer.sh ${device} ernie_crf > ${log_path}/python_infer_ernie_crf_${device}.log 2>&1
    print_info $? python_infer_ernie_crf_${device}
    bash python_infer.sh ${device} bigru_crf > ${log_path}/python_infer_bigru_crf_${device}.log 2>&1
    print_info $? python_infer_bigru_crf_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 2 msra_ner
msra_ner(){
# 固定接收5个参数，有扩展可以自定义
default_list="train,eval"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/msra_ner/
    cd ./msra_ner
    bash data_proc.sh
    cd ..
    MAX_STEPS=1000
    SAVE_STEPS=500
    LOGGING_STEPS=10
    MODEL_STEP=500
else
    log_path=$log_path
    MAX_STEPS=2
    SAVE_STEPS=2
    LOGGING_STEPS=1
    MODEL_STEP=2
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
# 单卡训练
train(){
    cd ./msra_ner
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} single ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} $LOGGING_STEPS >$log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    # 多卡训练
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} multi ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} $LOGGING_STEPS >$log_path/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd ..
}
## eval
eval(){
    cd ./msra_ner
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} ${MODEL_STEP} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    bash eval.sh ${device} ${MODEL_STEP} > $log_path/eval_${device}.log 2>&1
    print_info $? eval_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 3 glue
glue() {
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/glue/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./glue
    # 单卡
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' 'bert' 'bert-base-uncased' 'SST-2' 1e-4 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_bert-base-uncased_SST-2_single_${device}.log 2>&1
    print_info $? train_bert-base-uncased_SST-2_single_${device}
    #多卡
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash train.sh ${device} 'multi' 'bert' 'bert-base-uncased' 'SST-2' 1e-4 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_bert-base-uncased_SST-2_multi_${device}.log 2>&1
    print_info $? train_bert-base-uncased_SST-2_single_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 4 bert
# 5 skep (max save 不可控 内置)
# 6 bigbird
bigbird(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/bigbird/
    cd ./bigbird
    bash data_proc.sh
    cd ..
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
    default_list="train,finetune"
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    default_list="train"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./bigbird
    if [[ ${system} == "mac" ]];then
        bash train.sh ${device} 'bigbird-base-uncased' 'mac' 'None' ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_mac_${device}.log 2>&1
        print_info $? train_mac_${device}
    else
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'bigbird-base-uncased' 'single' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'bigbird-base-uncased' 'multi' ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    fi
    cd ..
}
finetune(){
    cd ./bigbird
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash finetune.sh ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_${device}.log 2>&1
    print_info $? finetune_${device}
    cd ..
}
glue(){
    cd ./bigbird
    unset CUDA_VISIBLE_DEVICES
    bash run_glue.sh ${device} ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/run_glue_${device}.log 2>&1
    print_info $? run_glue_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 7 electra
electra(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/electra/
    MAX_STEPS=12
    SAVE_STEPS=10
    LOGGING_STEPS=1
    default_list="train,finetune,infer"
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    default_list="train"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./electra
    bash data_proc.sh
    # 单卡训练
    export DATA_DIR=./BookCorpus/
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh 'single' ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/single_card_train.log 2>&1
    print_info $? single_card_train
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash train.sh 'multi' ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/multi_cards_train.log 2>&1
    print_info $? multi_cards_train
    cd ..
}
finetune(){
    cd ./electra
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash finetune.sh > $log_path/single_fine-tune.log 2>&1
    print_info $? single_fine-tune
    cd ..
}
infer(){
    cd ./electra
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh > $log_path/infer.log 2>&1
    print_info $? infer
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 9 ernie-1.0
ernie-1.0(){
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/language_model_ernie/
    MAX_STEPS=100
    SAVE_STEPS=20
else
    log_path=$log_path
    MAX_STEPS=40
    SAVE_STEPS=20
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./language_model_ernie
    # 收敛性case
    if [[ ${mode_tag} == "CE_CON" ]];then
        # 收敛性的数据集准备TODO
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} "multi" ${cudaid2} con > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    else
        # 获取数据
        bash data_proc.sh
        # 单卡训练
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} "single" ${cudaid1} common ${MAX_STEPS} ${SAVE_STEPS} >$log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
        bash train.sh ${device} "multi" ${cudaid2} common ${MAX_STEPS} ${SAVE_STEPS} >$log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 10 xlnet
xlnet(){
# 固定接收5个参数，有扩展可以自定义
default_list="finetune"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/xlnet/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
finetune(){
    cd ./xlnet
    unset CUDA_VISIBLE_DEVICES
    bash fine_tune.sh 'SST-2' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/SST-2-fine_tune.log 2>&1
    print_info $? SST-2-fine_tune
    if [[ ${mode_tag} == "CE" ]];then
        bash fine_tune.sh 'CoLA' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/CoLA-fine_tune.log 2>&1
        print_info $? CoLA-fine_tune
        bash fine_tune.sh 'MRPC' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/MRPC-fine_tune.log 2>&1
        print_info $? MRPC-fine_tune
        bash fine_tune.sh 'STS-B' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/STS-B-fine_tune.log 2>&1
        print_info $? STS-B-fine_tune
        bash fine_tune.sh 'QNLI' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/QNLI-fine_tune.log 2>&1
        print_info $? QNLI-fine_tune
        bash fine_tune.sh 'QQP' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/QQP-fine_tune.log 2>&1
        print_info $? QQP-fine_tune
        bash fine_tune.sh 'MNLI' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/MNLI-fine_tune.log 2>&1
        print_info $? MNLI-fine_tune
        bash fine_tune.sh 'RTE' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/RTE-fine_tune.log 2>&1
        print_info $? RTE-fine_tune
        bash fine_tune.sh 'WNLI' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/WNLI-fine_tune.log 2>&1
        print_info $? WNLI-fine_tune
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 11 ofa
ofa(){
# 固定接收5个参数，有扩展可以自定义
default_list="finetune,train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/ofa/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
    MODEL_STEP=10
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    MODEL_STEP=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
finetune(){
    cd ./ofa
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash finetune.sh ${device} 'single' SST-2 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_SST-2_single_${device}.log 2>&1
    print_info $? finetune_SST-2_single_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash finetune.sh ${device} 'multi' SST-2  ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_SST-2_multi_${device}.log 2>&1
    print_info $? finetune_SST-2_multi_${device}
    if [[ ${mode_tag} == "CE" ]];then
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' QNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QNLI_single_${device}.log 2>&1
        print_info $? finetune_QNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' QNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QNLI_multi_${device}.log 2>&1
        print_info $? finetune_QNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' CoLA ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_CoLA_single_${device}.log 2>&1
        print_info $? finetune_CoLA_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' CoLA ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_CoLA_multi_${device}.log 2>&1
        print_info $? finetune_CoLA_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' MRPC ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MRPC_single_${device}.log 2>&1
        print_info $? finetune_MRPC_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' MRPC ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MRPC_multi_${device}.log 2>&1
        print_info $? finetune_MRPC_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' STS-B ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_STS-B_single_${device}.log 2>&1
        print_info $? finetune_STS-B_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' STS-B ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_STS-B_multi_${device}.log 2>&1
        print_info $? finetune_STS-B_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' QQP ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QQP_single_${device}.log 2>&1
        print_info $? finetune_QQP_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' QQP ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QQP_multi_${device}.log 2>&1
        print_info $? finetune_QQP_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' MNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MNLI_single_${device}.log 2>&1
        print_info $? finetune_MNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' MNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MNLI_multi_${device}.log 2>&1
        print_info $? finetune_MNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' RTE ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_RTE_single_${device}.log 2>&1
        print_info $? finetune_RTE_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' RTE ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_RTE_multi_${device}.log 2>&1
        print_info $? finetune_RTE_multi_${device}
    fi
    cd ..
}
train(){
    cd ./ofa
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' SST-2 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_SST-2_single_${device}.log 2>&1
    print_info $? train_SST-2_single_${device}
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} 'multi' SST-2 ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_SST-2_multi_${device}.log 2>&1
    print_info $? train_SST-2_multi_${device}
    if [[ ${mode_tag} == "CE" ]];then
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' QNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QNLI_single_${device}.log 2>&1
        print_info $? train_QNLI_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' QNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QNLI_multi_${device}.log 2>&1
        print_info $? train_QNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' CoLA ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_CoLA_single_${device}.log 2>&1
        print_info $? train_CoLA_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' CoLA ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_CoLA_multi_${device}.log 2>&1
        print_info $? train_CoLA_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' MRPC ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MRPC_single_${device}.log 2>&1
        print_info $? train_MRPC_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' MRPC ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MRPC_multi_${device}.log 2>&1
        print_info $? train_MRPC_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' STS-B ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_STS-B_single_${device}.log 2>&1
        print_info $? train_STS-B_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' STS-B ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_STS-B_multi_${device}.log 2>&1
        print_info $? train_STS-B_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' QQP ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QQP_single_${device}.log 2>&1
        print_info $? train_QQP_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' QQP ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QQP_multi_${device}.log 2>&1
        print_info $? train_QQP_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' MNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MNLI_single_${device}.log 2>&1
        print_info $? train_MNLI_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' MNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MNLI_multi_${device}.log 2>&1
        print_info $? train_MNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' RTE ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_RTE_single_${device}.log 2>&1
        print_info $? train_RTE_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' RTE ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_RTE_multi_${device}.log 2>&1
        print_info $? train_RTE_multi_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 12 albert
albert(){
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/glue/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./glue
    # 单卡
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' 'albert' 'albert-base-v2' 'SST-2' 1e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_SST-2_single_${device}.log 2>&1
    print_info $? train_albert-base-v2_SST-2_single_${device}
    #多卡
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash train.sh ${device} 'multi' 'albert' 'albert-base-v2' 'SST-2' 1e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_SST-2_multi_${device}.log 2>&1
    print_info $? train_albert-base-v2_SST-2_multi_${device}
    if [[ ${mode_tag} == "CE" ]];then
        # CE覆盖的多个数据集
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' 'albert' 'albert-base-v2' 'MNLI' 3e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_MNLI_single_${device}.log 2>&1
        print_info $? train_albert-base-v2_MNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash train.sh ${device} 'multi' 'albert' 'albert-base-v2' 'MNLI' 3e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_MNLI_multi_${device}.log 2>&1
        print_info $? train_albert-base-v2_MNLI_multi_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 13 squad
squad(){
# 固定接收5个参数，有扩展可以自定义
default_list="train,dy_to_st_infer"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/SQuAD/
    MAX_STEPS=30
    SAVE_STEPS=10
    LOGGING_STEPS=10
    MODEL_STEP=10
    cd ./SQuAD
    bash data_proc.sh
    cd ..
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    MODEL_STEP=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./SQuAD
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} 1.1 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} >$log_path/train_${device}_1.1.log 2>&1
    print_info $? train_${device}_1.1
    if [[ ${mode_tag} == "CE" ]];then
        bash train.sh ${device} 2.0 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} >$log_path/train_${device}_2.0.log 2>&1
        print_info $? train_${device}_2.0
    fi
    cd ..
}
dy_to_st_infer(){
    # 动转静预测
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    cur_path=`pwd`
    cd ${nlp_dir}/examples/machine_reading_comprehension/SQuAD/
    python -u ./export_model.py \
        --model_type bert \
        --model_path ./tmp/squad/model_${MODEL_STEP}/ \
        --output_path ./infer_model/model >> ${log_path}/python_infer_${device}.log 2>&1
        print_info $? python_infer_${device}
    python -u deploy/python/predict.py \
        --model_type bert \
        --model_name_or_path ./infer_model/model \
        --batch_size 2 \
        --max_seq_length 384 >> ${log_path}/python_infer_${device}.log 2>&1
        print_info $? python_infer_${device}
    # 回到原来的路径
    cd $cur_path
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 14 tinybert
tinybert() {
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/tinybert/
    MAX_STEPS=30
    SAVE_STEPS=10
    LOGGING_STEPS=1
    MODEL_STEP=10
    teacher_path=""
    default_list="finetune,distill"
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    MODEL_STEP=1
    cp -r /ssd1/paddlenlp/download/tinybert/pretrained_models/ ${nlp_dir}/model_zoo/tinybert/
    teacher_path=./pretrained_models/SST-2/best_model_610/
    student_name=./tmp/SST-2/single/intermediate_distill_model_final.pdparams
    default_list="distill"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
finetune(){
    cd ./tinybert
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' SST-2 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_SST-2_single_${device}.log 2>&1
        print_info $? finetune_SST-2_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' SST-2 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_SST-2_multi_${device}.log 2>&1
        print_info $? finetune_SST-2_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' QNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QNLI_single_${device}.log 2>&1
        print_info $? finetune_QNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' QNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QNLI_multi_${device}.log 2>&1
        print_info $? finetune_QNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' CoLA ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_CoLA_single_${device}.log 2>&1
        print_info $? finetune_CoLA_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' CoLA ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_CoLA_multi_${device}.log 2>&1
        print_info $? finetune_CoLA_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' MRPC ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MRPC_single_${device}.log 2>&1
        print_info $? finetune_MRPC_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' MRPC ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MRPC_multi_${device}.log 2>&1
        print_info $? finetune_MRPC_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' QQP ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QQP_single_${device}.log 2>&1
        print_info $? finetune_QQP_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' QQP ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QQP_multi_${device}.log 2>&1
        print_info $? finetune_QQP_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' MNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MNLI_single_${device}.log 2>&1
        print_info $? finetune_MNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' MNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MNLI_multi_${device}.log 2>&1
        print_info $? finetune_MNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' RTE ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_RTE_single_${device}.log 2>&1
        print_info $? finetune_RTE_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' RTE ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_RTE_multi_${device}.log 2>&1
        print_info $? finetune_RTE_multi_${device}
    cd ..
}
distill(){
    #蒸馏，目前只跑单卡
    cd ./tinybert
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'single' SST-2 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_SST-2_single_${device}.log
        print_info $? distill_SST-2_single_${device}
        # export CUDA_VISIBLE_DEVICES=${cudaid2}
        # bash distill.sh ${device} 'multi' SST-2 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_SST-2_multi_${device}.log
        # print_info $? distill_SST-2_multi_${device}
    if [[ ${mode_tag} == "CE" ]];then
        # CE 继续覆盖其他数据集
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'single' QNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_QNLI_single_${device}.log
        print_info $? distill_QNLI_single_${device}
        # export CUDA_VISIBLE_DEVICES=${cudaid2}
        # bash distill.sh ${device} 'multi' QNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_QNLI_multi_${device}.log
        # print_info $? distill_QNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'single' CoLA ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_CoLA_single_${device}.log
        print_info $? distill_CoLA_single_${device}
        # export CUDA_VISIBLE_DEVICES=${cudaid2}
        # bash distill.sh ${device} 'multi' CoLA ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_CoLA_multi_${device}.log
        # print_info $? distill_CoLA_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'single' MRPC ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_MRPC_single_${device}.log
        print_info $? distill_MRPC_single_${device}
        # export CUDA_VISIBLE_DEVICES=${cudaid2}
        # bash distill.sh ${device} 'multi' MRPC ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_MRPC_multi_${device}.log
        # print_info $? distill_MRPC_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'single' QQP ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_QQP_single_${device}.log
        print_info $? distill_QQP_single_${device}
        # export CUDA_VISIBLE_DEVICES=${cudaid2}
        # bash distill.sh ${device} 'multi' QQP ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_QQP_multi_${device}.log
        # print_info $? distill_QQP_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'single' MNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_MNLI_single_${device}.log
        print_info $? distill_MNLI_single_${device}
        # export CUDA_VISIBLE_DEVICES=${cudaid2}
        # bash distill.sh ${device} 'multi' MNLI ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_MNLI_multi_${device}.log
        # print_info $? distill_MNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'single' RTE ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_RTE_single_${device}.log
        print_info $? distill_RTE_single_${device}
        # export CUDA_VISIBLE_DEVICES=${cudaid2}
        # bash distill.sh ${device} 'multi' RTE ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} > $log_path/distill_RTE_multi_${device}.log
        # print_info $? distill_RTE_multi_${device}
    else
        #预测层蒸馏;再用中间的蒸馏结果再做一次蒸馏
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash distill.sh ${device} 'predslim' SST-2 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${teacher_path} ${student_name} > $log_path/distill_SST-2_predslim_${device}.log
        print_info $? distill_SST-2_predslim_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 15 lexical_analysis
lexical_analysis(){
# 固定接收5个参数，有扩展可以自定义
default_list="train,infer,eval,dy_to_st_infer"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/lexical_analysis/
    EPOCHS=10
    SAVE_STEPS=100
    LOGGING_STEPS=10
    MODEL_STEP=100
else
    log_path=$log_path
    EPOCHS=1
    SAVE_STEPS=15
    LOGGING_STEPS=1
    MODEL_STEP=15
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./lexical_analysis
    bash data_proc.sh
    if [[ ${system} == "mac" ]];then
        bash train.sh ${device} "single" ${cudaid1} ${EPOCHS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
    else
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} "single" ${cudaid1} ${EPOCHS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} "multi" ${cudaid2} ${EPOCHS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    fi
    cd ..
}
infer(){
    cd ./lexical_analysis
    if [[ ${device} == "gpu" ]];then
        # 如果是gpu则设置下显卡
        export CUDA_VISIBLE_DEVICES=${cudaid1}
    fi
    bash infer.sh ${device} ${MODEL_STEP} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    cd ..
}
eval(){
    cd ./lexical_analysis
    if [[ ${device} == "gpu" ]];then
        # 如果是gpu则设置下显卡
        export CUDA_VISIBLE_DEVICES=${cudaid1}
    fi
    bash eval.sh ${device} ${MODEL_STEP} > $log_path/eval_${device}.log 2>&1
    print_info $? eval_${device}
    cd ..
}
dy_to_st_infer(){
    # CI有的CE没加
    cur_path=`pwd`
    cd ${nlp_dir}/examples/lexical_analysis/
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    python export_model.py \
        --data_dir=./lexical_analysis_dataset_tiny \
        --params_path=./save_dir/model_${MODEL_STEP}.pdparams \
        --output_path=./infer_model/static_graph_params >> ${log_path}/python_infer_single_${device}.log 2>&1
    print_info $? python_infer_single_${device}
    # deploy
    python deploy/predict.py \
    --model_file=infer_model/static_graph_params.pdmodel \
    --params_file=infer_model/static_graph_params.pdiparams \
    --data_dir lexical_analysis_dataset_tiny >> ${log_path}/python_infer_single_${device}.log 2>&1
    print_info $? python_infer_single_${device}
    cd $cur_path
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 16 seq2seq
seq2seq(){
# 固定接收5个参数，有扩展可以自定义
default_list="train,infer,dy_to_st_infer"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/seq2seq/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./seq2seq
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} "single" > $log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid2} 
    bash train.sh ${device} "multi" > $log_path/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd ..
}
infer(){
    cd ./seq2seq
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    cd ..
}
dy_to_st_infer(){
    cd ./seq2seq
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash InferFram.sh ${device} > $log_path/inferfram_${device}.log 2>&1
    print_info $? inferfram_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 17 pretrained_models
pretrained_models() {
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/text_classification_pretrained/
    default_list="train,infer,dy_to_st_infer"
else
    log_path=$log_path
    default_list="train,dy_to_st_infer"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./text_classification_pretrained
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} "single" ${cudaid1} > $log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} "multi" ${cudaid2} > $log_path/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd ..
}
infer(){
    cd ./text_classification_pretrained
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} "single" > $log_path/infer_single_${device}.log 2>&1
    print_info $? infer_single_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash infer.sh ${device} "multi" > $log_path/infer_multi_${device}.log 2>&1
    print_info $? infer_multi_${device}
    cd ..
}
dy_to_st_infer(){
    cur_path=`pwd`
    cd ${nlp_dir}/examples/text_classification/pretrained_models/
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    python export_model.py --params_path=./checkpoints/model_100/model_state.pdparams --output_path=./output >> ${log_path}/python_infer_${device}.log
    python deploy/python/predict.py --model_dir=./output >> ${log_path}/python_infer_${device}.log
    print_info $? python_infer_${device}
    cd $cur_path
}

if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 18 word_embedding 5min
word_embedding(){
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/word_embedding/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./word_embedding
    if [[ ${system} == "mac" ]];then
        bash train.sh ${device} single > $log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
    else
        # 交叉覆盖
        # 使用paddlenlp.embeddings.TokenEmbedding
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} single True > $log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
        # 使用paddle.nn.Embedding
        export CUDA_VISIBLE_DEVICES=${cudaid2} 
        bash train.sh ${device} multi > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 19 ernie-ctm
ernie-ctm(){
# 固定接收5个参数，有扩展可以自定义
default_list="train,infer"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/ernie-ctm/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./ernie-ctm
    bash data_proc.sh
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} 'single' ${cudaid1} > $log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    cd ..
}
infer(){
    cd ./ernie-ctm
    unset CUDA_VISIBLE_DEVICES
    bash infer.sh ${device} ${cudaid1} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 20 distilbert
distilbert(){
# 固定接收5个参数，有扩展可以自定义
default_list="train,distill"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
DATA_PATH='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt'
DATA_PATH_AMC='/Users/paddle/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt'
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/distill_lstm/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./distill_lstm
    bash data_proc.sh
    if [[ ${system} == "mac" ]];then
        bash train.sh ${device} mac chnsenticorp >$log_path/train_chnsenticorp_mac_${device}.log 2>&1
        print_info $? train_chnsenticorp_mac_${device}
        bash train.sh ${device} mac sst-2 ${DATA_PATH_AMC} >$log_path/train_sst-2_mac_${device}.log 2>&1
        print_info $? train_sst-2_mac_${device}
        bash train.sh ${device} mac qqp ${DATA_PATH_AMC} >$log_path/train_qqp_mac_${device}.log 2>&1
        print_info $? train_qqp_mac_${device}
    else
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} single sst-2 ${DATA_PATH} >$log_path/train_sst-2_single_${device}.log 2>&1
        print_info $? train_sst-2_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash train.sh ${device} multi sst-2 ${DATA_PATH} >$log_path/train_sst-2_multi_${device}.log 2>&1
        print_info $? train_sst-2_multi_${device}
        if [[ ${mode_tag} == "CE" ]];then
            export CUDA_VISIBLE_DEVICES=${cudaid1}
            bash train.sh ${device} single chnsenticorp >$log_path/train_chnsenticorp_single_${device}.log 2>&1
            print_info $? train_chnsenticorp_single_${device}
            export CUDA_VISIBLE_DEVICES=${cudaid2}
            bash train.sh ${device} multi chnsenticorp >$log_path/train_chnsenticorp_multi_${device}.log 2>&1
            print_info $? train_chnsenticorp_multi_${device}
            export CUDA_VISIBLE_DEVICES=${cudaid1}
            bash train.sh ${device} single qqp ${DATA_PATH} >$log_path/train_qqp_single_${device}.log 2>&1
            print_info $? train_qqp_single_${device}
            export CUDA_VISIBLE_DEVICES=${cudaid2}
            bash train.sh ${device} multi qqp ${DATA_PATH} >$log_path/train_qqp_multi_${device}.log 2>&1
            print_info $? train_qqp_multi_${device}
        fi
    fi
    cd ..
}
distill(){
    cd ./distill_lstm
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash distill.sh ${device} single sst-2 >$log_path/distill_sst-2_single_${device}.log
    print_info $? distill_sst-2_single_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash distill.sh ${device} multi sst-2 >$log_path/distill_sst-2_multi_${device}.log
    print_info $? distill_sst-2_multi_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 21 stacl
stacl(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    default_list="train,infer"
    log_path=${log_path}/stacl/
    EPOCHS=1
    MAX_STEPS=100
    SAVE_STEPS=10
    LOGGING_STEPS=10
else
    default_list="train,infer,recv_train"
    log_path=$log_path
    EPOCHS=1
    MAX_STEPS=3
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./stacl
    bash data_proc.sh
    if [[ ${mode_tag} == "CE_CON" ]];then
        # 收敛性的数据集准备TODO
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} "multi" ${cudaid2} con > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    else
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} "single" ${cudaid1} common ${EPOCHS} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
        bash train.sh ${device} "multi" ${cudaid2} common ${EPOCHS} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    fi
    cd ..
}
infer(){
    cd ./stacl
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} "single" > $log_path/infer_single_${device}.log 2>&1
    print_info $? infer_single_${device}
    cd ..
}
recv_train(){
    # 恢复训练
    cur_path=`pwd`
    cd ${nlp_dir}/examples/simultaneous_translation/stacl
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    sed -i "s/waitk: -1/waitk: 3/g" config/transformer.yaml
    sed -i 's/save_model: "trained_models"/save_model: "trained_models_3"/g' config/transformer.yaml
    sed -i 's#init_from_checkpoint: ""#init_from_checkpoint: "./trained_models/step_1/"#g' config/transformer.yaml
    python -m paddle.distributed.launch  train.py --config ./config/transformer.yaml >${log_path}/stacl_wk3.log 2>&1
    print_info $? stacl_wk3
    sed -i "s/waitk: 3/waitk: 5/g" config/transformer.yaml
    sed -i 's/save_model: "trained_models_3"/save_model: "trained_models_5"/g' config/transformer.yaml
    sed -i 's#init_from_checkpoint: "./trained_models/step_1/"#init_from_checkpoint: "./trained_models_3/step_1/"#g' config/transformer.yaml
    python -m paddle.distributed.launch train.py --config ./config/transformer.yaml >${log_path}/stacl_wk5.log 2>&1
    print_info $? stacl_wk5
    cd $cur_path
}

if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 22 transformer
#确认下数据集

# 23 pet
pet(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train,infer"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/few_shot_pet/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./few_shot_pet
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} 'single' ${cudaid1} chid 0 > ${log_path}/train_chid_single_${device}.log 2>&1
    print_info $? train_chid_single_${device}
    # CE 覆盖更多数据集
    if [[ ${mode_tag} == "CE" ]];then
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'single' ${cudaid1} iflytek 0 > ${log_path}/train_iflytek_single_${device}.log 2>&1
        print_info $? train_iflytek_single_${device}
        bash train.sh ${device} 'single' ${cudaid1} tnews 0.5 > ${log_path}/train_tnews_single_${device}.log 2>&1
        print_info $? train_tnews_single_${device}
        bash train.sh ${device} 'single' ${cudaid1} eprstmt 0 > ${log_path}/train_eprstmt_single_${device}.log 2>&1
        print_info $? train_eprstmt_single_${device}
        bash train.sh ${device} 'single' ${cudaid1} bustm 0 > ${log_path}/train_bustm_single_${device}.log 2>&1
        print_info $? train_bustm_single_${device}
        bash train.sh ${device} 'single' ${cudaid1} ocnli 0 > ${log_path}/train_ocnli_single_${device}.log 2>&1
        print_info $? train_ocnli_single_${device}
        bash train.sh ${device} 'single' ${cudaid1} csl 0 > ${log_path}/train_csl_single_${device}.log 2>&1
        print_info $? train_csl_single_${device}
        bash train.sh ${device} 'single' ${cudaid1} csldcp 0 > ${log_path}/train_csldcp_single_${device}.log 2>&1
        print_info $? train_csldcp_single_${device}
        bash train.sh ${device} 'single' ${cudaid1} cluewsc 0 > ${log_path}/train_cluewsc_single_${device}.log 2>&1
        print_info $? train_cluewsc_single_${device}
    fi
    cd ..
}
infer(){
    cd ./few_shot_pet
    unset CUDA_VISIBLE_DEVICES
    bash infer.sh ${device} 'single' ${cudaid1} chid 3 > $log_path/infer_chid_single_${device}.log 2>&1
    print_info $? infer_chid_single_${device}
    # CE 覆盖更多数据集
    if [[ ${mode_tag} == "CE" ]];then
        unset CUDA_VISIBLE_DEVICES
        bash infer.sh ${device} 'single' ${cudaid1} iflytek 58 > $log_path/infer_iflytek_single_${device}.log 2>&1
        print_info $? infer_iflytek_single_${device}
        bash infer.sh ${device} 'single' ${cudaid1} tnews 15 > $log_path/infer_tnews_single_${device}.log 2>&1
        print_info $? infer_tnews_single_${device}
        bash infer.sh ${device} 'single' ${cudaid1} eprstmt 2 > $log_path/infer_eprstmt_single_${device}.log 2>&1
        print_info $? infer_eprstmt_single_${device}
        bash infer.sh ${device} 'single' ${cudaid1} bustm 2 > $log_path/infer_bustm_single_${device}.log 2>&1
        print_info $? infer_bustm_single_${device}
        bash infer.sh ${device} 'single' ${cudaid1} ocnli 2 > $log_path/infer_ocnli_single_${device}.log 2>&1
        print_info $? infer_ocnli_single_${device}
        bash infer.sh ${device} 'single' ${cudaid1} csl 2 > $log_path/infer_csl_single_${device}.log 2>&1
        print_info $? infer_csl_single_${device}
        bash infer.sh ${device} 'single' ${cudaid1} csldcp 34 > $log_path/infer_csldcp_single_${device}.log 2>&1
        print_info $? infer_csldcp_single_${device}
        bash infer.sh ${device} 'single' ${cudaid1} cluewsc 2 > $log_path/infer_cluewsc_single_${device}.log 2>&1
        print_info $? infer_cluewsc_single_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#24 simbert
simbert(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="infer"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/simbert/
    data_path=/workspace/task/datasets/simbert
else
    log_path=$log_path
    data_path=/ssd1/paddlenlp/download/simbert
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

infer(){
    cd ./simbert
    bash data_proc.sh ${data_path}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} 'single' > ${log_path}/infer_single_${device}.log 2>&1
    print_info $? infer_single_${device}
    cd ..
}

if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#25 ernie-doc
ernie-doc(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/language_model_ernie_doc/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./language_model_ernie_doc
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash classifier.sh ${device} "single" ${cudaid1} ernie-doc-base-en hyp > $log_path/train_single_hyp_${device}.log 2>&1
    print_info $? train_single_hyp_${device}
    unset CUDA_VISIBLE_DEVICES
    bash classifier.sh ${device} "multi" ${cudaid2} ernie-doc-base-en hyp > $log_path/train_multi_hyp_${device}.log 2>&1
    print_info $? train_multi_hyp_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash reading_cmp.sh ${device} "single" ${cudaid1}  ernie-doc-base-zh cmrc2018 > $log_path/train_single_cmrc2018_${device}.log 2>&1
    print_info $? train_single_cmrc2018_${device}
    unset CUDA_VISIBLE_DEVICES
    bash reading_cmp.sh ${device} "multi" ${cudaid2} ernie-doc-base-zh cmrc2018  > $log_path/train_multi_cmrc2018_${device}.log 2>&1
    print_info $? train_multi_cmrc2018_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash single_choice.sh ${device} "single" ${cudaid1} ernie-doc-base-zh c3  > $log_path/train_single_c3_${device}.log 2>&1
    print_info $? train_single_c3_${device}
    unset CUDA_VISIBLE_DEVICES
    bash single_choice.sh ${device} "multi" ${cudaid2}  ernie-doc-base-zh c3  > $log_path/train_multi_c3_${device}.log 2>&1
    print_info $? train_multi_c3_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash semantic.sh ${device} "single" ${cudaid1} ernie-doc-base-zh cail2019_scm > $log_path/train_single_cail2019_scm_${device}.log 2>&1
    print_info $? train_single_cail2019_scm_${device}
    unset CUDA_VISIBLE_DEVICES
    bash semantic.sh ${device} "multi" ${cudaid2}  ernie-doc-base-zh cail2019_scm  > $log_path/train_multi_cail2019_scm_${device}.log 2>&1
    print_info $? train_multi_cail2019_scm_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash sequence.sh ${device} "single" ${cudaid1} ernie-doc-base-zh msra_ner > $log_path/train_single_msra_ner_${device}.log 2>&1
    print_info $? train_single_msra_ner_${device}
    unset CUDA_VISIBLE_DEVICES
    bash sequence.sh ${device} "multi" ${cudaid2}  ernie-doc-base-zh msra_ner > $log_path/train_multi_msra_ner_${device}.log 2>&1
    print_info $? train_multi_msra_ner_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash reading_cmp.sh ${device} "single" ${cudaid1} ernie-doc-base-zh dureader_robust > $log_path/train_single_dureader_robust_${device}.log 2>&1
    print_info $? train_single_dureader_robust_${device}
    unset CUDA_VISIBLE_DEVICES
    bash reading_cmp.sh ${device} "multi" ${cudaid2}  ernie-doc-base-zh dureader_robust > $log_path/train_multi_dureader_robust_${device}.log 2>&1
    print_info $? train_multi_dureader_robust_${device}
    if [[ ${mode_tag} == "CE" ]];then
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash classifier.sh ${device} "single" ${cudaid1} ernie-doc-base-en imdb > $log_path/train_single_imdb_${device}.log 2>&1
        print_info $? train_single_imdb_${device}
        unset CUDA_VISIBLE_DEVICES
        bash classifier.sh ${device} "multi" ${cudaid2} ernie-doc-base-en imdb > $log_path/train_multi_imdb_${device}.log 2>&1
        print_info $? train_multi_imdb_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash classifier.sh ${device} "single" ${cudaid1} ernie-doc-base-zh iflytek > $log_path/train_single_iflytek_${device}.log 2>&1
        print_info $? train_single_iflytek_${device}
        unset CUDA_VISIBLE_DEVICES
        bash classifier.sh ${device} "multi" ${cudaid2} ernie-doc-base-zh iflytek > $log_path/train_multi_iflytek_${device}.log 2>&1
        print_info $? train_multi_iflytek_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash classifier.sh ${device} "single" ${cudaid1} ernie-doc-base-zh thucnews > $log_path/train_single_thucnews_${device}.log 2>&1
        print_info $? train_single_thucnews_${device}
        unset CUDA_VISIBLE_DEVICES
        bash classifier.sh ${device} "multi" ${cudaid2} ernie-doc-base-zh thucnews > $log_path/train_multi_thucnews_${device}.log 2>&1
        print_info $? train_multi_thucnews_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash reading_cmp.sh ${device} "single" ${cudaid1} ernie-doc-base-zh drcd  > $log_path/train_single_drcd_${device}.log 2>&1
        print_info $? train_single_drcd_${device}
        unset CUDA_VISIBLE_DEVICES
        bash reading_cmp.sh ${device} "multi" ${cudaid2} ernie-doc-base-zh drcd > $log_path/train_multi_drcd_${device}.log 2>&1
        print_info $? train_multi_drcd_${device}
    fi
    cd ..
}

if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done   
}

#26 transformer-xl
transformer-xl(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train,eval"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/transformer-xl/
    data_path=/workspace/task/datasets/transformer-xl
else
    log_path=$log_path
    data_path=/ssd1/paddlenlp/download/transformer-xl
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./transformer-xl
    bash data_proc.sh ${data_path}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh 'single' > $log_path/single_enwik8_train.log 2>&1
    print_info $? single_enwik8_train
    unset CUDA_VISIBLE_DEVICES
    bash train.sh 'multi' ${cudaid2} > $log_path/multi_enwik8_train.log 2>&1
    print_info $? multi_enwik8_train
    cd ..
}
eval(){
    cd ./transformer-xl
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash eval.sh > $log_path/enwiki_eval.log 2>&1
    print_info $? enwiki_eval
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#27 pointer_summarizer
pointer_summarizer(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/pointer_summarizer/
    data_path=/workspace/task/datasets/pointer_summarizer
    MAX_STEPS=30
else
    log_path=$log_path
    MAX_STEPS=5
    data_path=/ssd1/paddlenlp/download/pointer_summarizer
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./pointer_summarizer
    bash data_proc.sh ${data_path}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' ${MAX_STEPS} > $log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#28 question_matching
question_matching(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train,infer"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/text_matching_question_matching/
    data_path=/workspace/task/datasets/question_matching
    MAX_STEPS=100
    SAVE_STEPS=20
    EVAL_STEP=50
else
    log_path=$log_path
    data_path=/ssd1/paddlenlp/download/question_matching
    MAX_STEPS=10
    SAVE_STEPS=10
    EVAL_STEP=10
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./text_matching_question_matching
    bash data_proc.sh ${data_path}
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} single ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${EVAL_STEP} > $log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} multi ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${EVAL_STEP}  >$log_path/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd ..
}
infer(){
    cd ./text_matching_question_matching
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} ${EVAL_STEP} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 29 ernie-csc
ernie-csc(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train,infer,eval,dy_to_st_infer"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/ernie-csc/
    MAX_STEPS=200
    SAVE_STEPS=100
    LOGGING_STEPS=10
else
    log_path=$log_path
    MAX_STEPS=100
    SAVE_STEPS=100
    LOGGING_STEPS=10
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./ernie-csc
    bash data_proc.sh
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} 'multi' ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd ..
}
infer(){
    cd ./ernie-csc
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    cd ..
}
eval(){
    cd ./ernie-csc
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash eval.sh ${device} > $log_path/eval_${device}.log 2>&1
    print_info $? eval_${device}
    cd ..
}
dy_to_st_infer(){
    cd ./ernie-csc
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash deploy_infer.sh ${device} > $log_path/infer_deploy_${device}.log 2>&1
    print_info $? infer_deploy_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#30 nptag
nptag(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train,infer,dy_to_st_infer"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/nptag/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi  
train(){
    cd ./nptag
    bash data_proc.sh
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} 'single' ${cudaid1} > $log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    bash train.sh ${device} 'multi' ${cudaid2} > $log_path/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd ..
}
infer(){
    cd ./nptag
    unset CUDA_VISIBLE_DEVICES
    bash infer.sh ${device} ${cudaid1} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    cd ..
}
dy_to_st_infer(){
    cd ./nptag
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash python_infer.sh ${device} ${cudaid1} > $log_path/python_infer_${device}.log 2>&1
    print_info $? python_infer_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#31 ernie-m
ernie-m(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="train"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/ernie-m/
    MAX_STEPS=30
    SAVE_STEPS=10
    LOGGING_STEPS=10

else
    log_path=$log_path
    MAX_STEPS=2
    SAVE_STEPS=2
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cur_path=`pwd`
    cd ${nlp_dir}/model_zoo/ernie-m
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    python run_classifier.py \
        --task_type cross-lingual-transfer \
        --batch_size 8 \
        --model_name_or_path ernie-m-base \
        --save_steps ${SAVE_STEPS} \
        --max_steps ${MAX_STEPS} \
        --output_dir output \
        --logging_steps ${LOGGING_STEPS} >${log_path}/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    # 多卡
    unset CUDA_VISIBLE_DEVICES
    python -m paddle.distributed.launch --gpus ${cudaid2} --log_dir output run_classifier.py  \
        --task_type cross-lingual-transfer  \
        --batch_size 8    \
        --model_name_or_path ernie-m-base \
        --save_steps ${SAVE_STEPS} \
        --max_steps ${MAX_STEPS} \
        --output_dir output \
        --logging_steps ${LOGGING_STEPS} >${log_path}/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd $cur_path
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#32 clue
clue(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="finetune,classification,reading_cmp"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/clue/
    MAX_STEPS=30
    SAVE_STEPS=10
    LOGGING_STEPS=10

else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
finetune(){
    cur_path=`pwd`
    cd ${nlp_dir}/examples/benchmark/clue/classification
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    # 预训练
    python -u ./run_clue_classifier_trainer.py \
        --model_name_or_path ernie-3.0-base-zh \
        --dataset "clue afqmc" \
        --max_seq_length 128 \
        --per_device_train_batch_size 32   \
        --per_device_eval_batch_size 32   \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --logging_steps ${LOGGING_STEPS} \
        --seed 42  \
        --save_steps ${SAVE_STEPS} \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --adam_epsilon 1e-8 \
        --output_dir ./tmp \
        --device ${device}  \
        --do_train \
        --do_eval \
        --metric_for_best_model "eval_accuracy" \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --max_steps ${MAX_STEPS} >${log_path}/finetune_afqmc_single_${device}.log 2>&1
    print_info $? finetune_afqmc_single_${device}
    cd ${cur_path}
}
classification(){
    # 分类训练
    cur_path=`pwd`
    cd ${nlp_dir}/examples/benchmark/clue/classification
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    python -u run_clue_classifier.py  \
        --model_name_or_path ernie-3.0-base-zh \
        --task_name afqmc \
        --max_seq_length 128 \
        --batch_size 16   \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --logging_steps ${LOGGING_STEPS} \
        --seed 42  \
        --save_steps ${SAVE_STEPS} \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --adam_epsilon 1e-8 \
        --output_dir ./output/afqmc \
        --device gpu \
        --max_steps ${MAX_STEPS} \
        --do_train  >${log_path}/classification_afqmc_single_${device}.log 2>&1
    print_info $? classification_afqmc_single_${device}
    cd ${cur_path}
}
reading_cmp(){   
    cur_path=`pwd`
    cd ${nlp_dir}/examples/benchmark/clue/mrc
    unset CUDA_VISIBLE_DEVICES
    python -m paddle.distributed.launch --gpus ${cudaid2} run_c3.py \
        --model_name_or_path ernie-3.0-base-zh \
        --batch_size 6 \
        --learning_rate 2e-5 \
        --max_seq_length 512 \
        --num_train_epochs 2 \
        --do_train \
        --warmup_proportion 0.1 \
        --gradient_accumulation_steps 3 \
        --max_steps ${MAX_STEPS} \
        --output_dir ./tmp >${log_path}/reading_cmp_cmrc_single_${device}.log 2>&1
    print_info $? reading_cmp_cmrc_single_${device}
    cd ${cur_path}
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

#33 taskflow
taskflow(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
default_list="interact"
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/taskflow/
else
    log_path=$log_path
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
interact(){
    cd ./taskflow
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash interact.sh > $log_path/train_${device}.log 2>&1
    print_info $? train_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}




####################################
# 程序的调度入口
model_func=$1
args=""
if [ $# -gt 0 ]; then
    args=${@:2}
fi
echo $args
echo $model_func
${model_func} $args
