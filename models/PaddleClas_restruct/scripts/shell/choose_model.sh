echo $output_type
export params_dir=$(ls ${output_dir}/${model_name})
echo "######  params_dir"
echo $params_dir
if [[ ${predict_step} == "" ]];then     #要区分下不能把之前的训好的覆盖了
    if [[ ${output_type} == "trained" ]];then
        if [[ -f ${output_dir}/${model_name}/${params_dir}/latest.pdparams ]];then
            export pretrained_model=${output_dir}/${model_name}/${params_dir}/latest
        else
            export pretrained_model="None"  #使用初始化参数评估
        fi
    elif [[ ${output_type} == "pretrained" ]];then
        # PaddleClas/ppcls/arch/backbone/legendary_models/
        # esnet.py    ESNet
        # hrnet.py    HRNet
        # inception_v3.py     InceptionV3
        # mobilenet_v1.py     MobileNetV1
        # mobilenet_v3.py     MobileNetV3
        # pp_hgnet.py         PPHGNet
        # pp_lcnet.py         PPLCNet
        # pp_lcnet_v2.py      PPLCNetV2
        # resnet.py       ResNet
        # swin_transformer.py     SwinTransformer
        # vgg.py      VGG
        if [[ -f ${params_dir}_pretrained.pdparams ]];then #有下载好的跳过下载
            export pretrained_model=${params_dir}_pretrained
        else
            if [[ ${params_dir} =~ "ESNet" ]] || [[ ${params_dir} =~ "HRNet" ]] || [[ ${params_dir} =~ "InceptionV3" ]] || \
                [[ ${params_dir} =~ "MobileNetV1" ]] || [[ ${params_dir} =~ "MobileNetV3" ]] || [[ ${params_dir} =~ "PPHGNet" ]] || \
                [[ ${params_dir} =~ "PPLCNet" ]] || [[ ${params_dir} =~ "PPLCNetV2" ]] || [[ ${params_dir} =~ "ResNet" ]] || \
                [[ ${params_dir} =~ "SwinTransformer" ]] || [[ ${params_dir} =~ "VGG" ]];then
                echo "######  use legendary_models pretrain model"
                wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/${params_dir}_pretrained.pdparams --no-proxy
            else
                wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/${params_dir}_pretrained.pdparams --no-proxy
            fi
            if [[ $? -eq 0 ]];then
                export pretrained_model=${params_dir}_pretrained
            else
                echo "\033[31m pretrained download ${model_name}/${params_dir} failed!\033[0m"
                export pretrained_model=${output_dir}/${model_name}/${params_dir}/latest
            fi
        fi
        # 单独考虑
        # if [[ ${model} =~ 'distill_pphgnet_base' ]]  || [[ ${model} =~ 'PPHGNet_base' ]] ;then
        #     echo "######  use distill_pphgnet_base pretrain model"
        #     echo ${model}
        #     echo ${params_dir}
        #     wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_base_ssld_pretrained.pdparams --no-proxy
        #     rm -rf output/$params_dir/latest.pdparams
        #     \cp -r -f PPHGNet_base_ssld_pretrained.pdparams output/$params_dir/latest.pdparams
        #     rm -rf PPHGNet_base_ssld_pretrained_pretrained.pdparams
        # fi

        # if [[ ${model} =~ 'PPLCNet' ]]  && [[ ${model} =~ 'dml' ]] ;then #注意区分dml 与 udml
        #     echo "######  use PPLCNet dml pretrain model"
        #     echo ${model}
        #     echo ${params_dir}
        #     wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Distillation/${model}_pretrained.pdparams --no-proxy
        #     rm -rf output/$params_dir/latest.pdparams
        #     \cp -r -f ${model}_pretrained.pdparams output/$params_dir/latest.pdparams
        #     rm -rf ${model}_pretrained.pdparams
        # fi

    else
        export pretrained_model="None"  #使用初始化参数评估
    fi
    echo ${pretrained_model}
else
    if [[ ${output_type} == "trained" ]];then
        if [[ -f "inference/${model_name}/inference.pdmodel" ]];then
            export pretrained_model="../inference/${model_name}"
        else
            export pretrained_model="None" #必须有下好的模型，不能使用初始化模型，所以不管用默认参数还是None都不能预测
        fi
    elif [[ ${output_type} == "pretrained" ]];then
        if [[ -d ${params_dir}_infer ]] && [[ -f ${params_dir}_infer/inference.pdiparams ]];then #有下载好的，或者export已导出的跳过下载
            export pretrained_model="../${params_dir}_infer"
        else
            wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/${params_dir}_infer.tar --no-proxy
            if [[ $? -eq 0 ]];then
                tar xf ${params_dir}_infer.tar
                export pretrained_model="../${params_dir}_infer"
            else
                echo "\033[31m pretrained download ${model_name}/${params_dir} failed!\033[0m"
                export pretrained_model="../inference/${model_name}"
            fi
        fi
    else
        export pretrained_model="None"
    fi
fi
