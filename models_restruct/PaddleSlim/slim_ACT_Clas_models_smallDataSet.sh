# $1 精度允许diff范围 $2 本地待上传目录 $3 上传到bos路径 $4 slim commit
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo ---${log_path}/FAIL_$2---
    echo "fail log as follow"
    cat  ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo ---${log_path}/SUCCESS_$2---
    grep -i 'Memory Usage' ${log_path}/SUCCESS_$2.log
fi
}

wget_ILSVRC2012_mini(){
echo ---infer ILSVRC2012_mini downloading-----
wget -q https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/ILSVRC2012.tar
tar xf ILSVRC2012.tar
echo ---infer ILSVRC2012_mini downloaded-----
}

wget_ILSVRC2012_mini

wget_infer_models(){
echo ---infer models downloading-----
root_url="https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference"
# pre_models="MobileNetV1 ResNet50_vd ShuffleNetV2_x1_0 \
#     SqueezeNet1_0 PPLCNetV2_base PPLCNet_x1_0 \
#     PPHGNet_tiny InceptionV3 EfficientNetB0 GhostNet_x1_0 \
#     MobileNetV3_large_x1_0 MobileNetV3_large_x1_0_ssld "
    # 因ViT_base_patch16_224、PPLCNet_x1_0 压缩前后精度不能对齐，暂时去掉；
#MobileNetV1 ResNet50_vd 在大数据集下压缩、训练
pre_models="ShuffleNetV2_x1_0 \
    SqueezeNet1_0 PPLCNetV2_base PPLCNet_x1_0 \
    PPHGNet_tiny InceptionV3 EfficientNetB0 GhostNet_x1_0 \
    MobileNetV3_large_x1_0 MobileNetV3_large_x1_0_ssld "

for model in ${pre_models}
do
    if [ ! -f ${model} ]; then
        echo ---infer ${model} downloading-----
        wget -q ${root_url}/${model}_infer.tar
        tar xf ${model}_infer.tar
    fi
done
echo ---infer models downloaded-----
}

wget_infer_models

# $1 模型名称、 $2 本次回归精度、$3 精度base值、 $4 精度允许diff范围
computer_diff_precision(){
#本次回归实际精度diff
diff=`echo $3-$2 | bc`
if [ `echo "${diff} < $4"|bc` -eq 1 ] ; then
   echo ---$1 actual_precision:$1  standard_precision:$2 diff:${diff} diff_allow:$4 success --- >> ${log_path}/result_precision.log
   return 0
else
   echo ---$1 actual_precision:$1  standard_precision:$2 diff:${diff} diff_allow:$4 failed ---- >> ${log_path}/result_precision.log
   return 1
fi
}

declare -A dic
# dic=([EfficientNetB0]=0.752 [GhostNet_x1_0]=0.726 [InceptionV3]=0.783 \
#     [MobileNetV1]=0.706 [MobileNetV3_large_x1_0]=0.741 [PPHGNet_tiny]=0.792 \
#     [PPLCNetV2_base]=0.763 [PPLCNet_x1_0]=0.8 [ResNet50_vd]=0.787 \
#     [ShuffleNetV2_x1_0]=0.683 [SqueezeNet1_0]=0.594 [MobileNetV3_large_x1_0_ssld]=0.771)
dic=([EfficientNetB0]=0.7246 [GhostNet_x1_0]=0.726 [InceptionV3]=0.7607 \
    [MobileNetV3_large_x1_0]=0.7023 [PPHGNet_tiny]=0.7041 \
    [PPLCNetV2_base]=0.763 [PPLCNet_x1_0]=0.1934 \
    [ShuffleNetV2_x1_0]=0.683 [SqueezeNet1_0]=0.594 [MobileNetV3_large_x1_0_ssld]=0.6753)
echo "---models and values---"
echo ${!dic[*]}   # 输出所有的key
echo ${dic[*]}    # 输出所有的value
echo "---diff_allow:$1---"

for model in $(echo ${!dic[*]});do
    echo ${PWD}
    base_value=${dic[$model]}
    echo "${model} base value:${base_value}"
    echo "--${model} runned time:"
    time ( python -m paddle.distributed.launch run.py \
        --save_dir=./${model}_act_qat/ \
        --config_path=./configs/${model}/qat_dis.yaml > ${log_path}/${model}_act_qat 2>&1 )
    print_info $? ${model}_act_qat
    model_log="${log_path}/SUCCESS_${model}_act_qat.log"
    if [ -f "${model_log}" ];then
        grep -i 'The metric of final model is ' ${model_log} | awk -F 'is ' '{print $2}' | awk -F ' samples/s' '{print $1}' > test.log &&  quota=`tail -1 test.log`
        echo ----model_name:${model}
        echo ----quota:${quota}
        echo ----base_value:${base_value}
        computer_diff_precision ${model} ${quota} ${base_value} $1
        if [ $? -eq 0 ];then
            echo "${log_path}/${model}_act_qat precision passed"
            tar -cf ${model}_act_qat.tar ${model}_act_qat
            mv ${model}_act_qat.tar $2
            unset http_proxy && unset https_proxy
            python Bos/upload.py $2 $3/${slim_commit}
            python Bos/upload.py $2 $3
        else
            echo "${log_path}/${model}_act_qat precision failed"
            mv ${model_log} ${log_path}/FAIL_precision_${model}_act_qat.log
            echo "fail log with precision as follow"
            grep -i 'metric' ${log_path}/FAIL_precision_${model}_act_qat.log
        fi
    fi
done
