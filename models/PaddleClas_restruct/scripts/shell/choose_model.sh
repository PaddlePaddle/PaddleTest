
echo "#### input_model_type"
echo ${input_model_type}
export Project_path=${Project_path:-$PWD}

cd ${Project_path} #确定下执行路径

function download_infer_tar(){
    cd deploy
    if [[ ! -d "models" ]];then
        mkdir models
    fi
    cd models
    if [[ -f "${1}.tar" ]] && [[ -d "${1}" ]];then
        echo "already download ${1}"
    else
        wget -q -c  \
            https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/${1}.tar
        tar xf ${1}.tar
    fi
    if [[ $? -eq 0 ]];then
        echo -e "\033[31m successfully! predict pretrained download ${model_name}/${infer_pretrain} successfully!\033[0m"
    else
        echo -e "\033[31m failed! predict pretrained download ${model_name}/${infer_pretrain} failed!\033[0m"
    fi
    cd ../../
}

if [[ ${predict_step} == "" ]];then     #要区分下不能把之前的训好的覆盖了
    if [[ ${input_model_type} == "trained" ]];then
        if [[ -d ${output_dir}/${model_name} ]];then
            export pretrained_model=${output_dir}/${model_name}/${params_dir}/latest
        else
            export pretrained_model="None"  #使用初始化参数评估
        fi
    elif [[ ${input_model_type} == "pretrained" ]];then
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
        if [[ -f ${pdparams_pretrain}_pretrained.pdparams ]];then #有下载好的跳过下载
            export pretrained_model=${pdparams_pretrain}_pretrained
        else
            if [[ ${model_name} =~ "-ESNet" ]] || [[ ${model_name} =~ "-HRNet" ]] || [[ ${model_name} =~ "-InceptionV3" ]] || \
                [[ ${model_name} =~ "-MobileNetV1" ]] || [[ ${model_name} =~ "-MobileNetV3" ]] || [[ ${model_name} =~ "-PPHGNet" ]] || \
                [[ ${model_name} =~ "-PPLCNet" ]] || [[ ${model_name} =~ "-PPLCNetV2" ]] || [[ ${model_name} =~ "-ResNet" ]] || \
                [[ ${model_name} =~ "-GeneralRecognition_PPLCNet" ]] || [[ ${model_name} =~ "-SwinTransformer" ]] || [[ ${model_name} =~ "-VGG" ]];then
                echo "######  use legendary_models pretrain model"
                wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/${pdparams_pretrain}_pretrained.pdparams --no-proxy
            else
                wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/${pdparams_pretrain}_pretrained.pdparams --no-proxy
            fi
            if [[ $? -eq 0 ]];then
                export pretrained_model=${pdparams_pretrain}_pretrained
            else
                echo -e "\033[31m failed! eval pretrained download ${model_name}/${pdparams_pretrain} failed!\033[0m"
                export pretrained_model=${output_dir}/${model_name}/${params_dir}/latest
            fi
        fi
        # 单独考虑
        # if [[ ${model} =~ 'distill_pphgnet_base' ]]  || [[ ${model} =~ 'PPHGNet_base' ]] ;then
        #     echo "######  use distill_pphgnet_base pretrain model"
        #     echo ${model}
        #     echo ${pdparams_pretrain}
        #     wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_base_ssld_pretrained.pdparams --no-proxy
        #     rm -rf output/$pdparams_pretrain/latest.pdparams
        #     \cp -r -f PPHGNet_base_ssld_pretrained.pdparams output/$pdparams_pretrain/latest.pdparams
        #     rm -rf PPHGNet_base_ssld_pretrained_pretrained.pdparams
        # fi

        # if [[ ${model} =~ 'PPLCNet' ]]  && [[ ${model} =~ 'dml' ]] ;then #注意区分dml 与 udml
        #     echo "######  use PPLCNet dml pretrain model"
        #     echo ${model}
        #     echo ${pdparams_pretrain}
        #     wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Distillation/${model}_pretrained.pdparams --no-proxy
        #     rm -rf output/$pdparams_pretrain/latest.pdparams
        #     \cp -r -f ${model}_pretrained.pdparams output/$pdparams_pretrain/latest.pdparams
        #     rm -rf ${model}_pretrained.pdparams
        # fi

    else
        export pretrained_model="None"  #使用初始化参数评估
    fi
else
    if [[ ${input_model_type} == "trained" ]];then #用训好的模型

        case ${model_type} in
        ImageNet|slim|PULC|DeepHash)
            if [[ -d "inference/${model_name}" ]];then
                export pretrained_model="../inference/${model_name}"
            else
                export pretrained_model="None" #必须有下好的模型，不能使用初始化模型，所以不管用默认参数还是None都不能预测
            fi
        ;;
        GeneralRecognition) #暂时用训好的模型 220815
            download_infer_tar picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer
            download_infer_tar general_PPLCNet_x2_5_lite_v1.0_infer
        ;;
        Cartoonface)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar cartoon_rec_ResNet50_iCartoon_v1.0_infer
        ;;
        Logo)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar logo_rec_ResNet50_Logo3K_v1.0_infer
        ;;
        Products)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar product_ResNet50_vd_aliproduct_v1.0_infer
        ;;
        Vehicle)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar vehicle_cls_ResNet50_CompCars_v1.0_infer
        ;;
        reid|metric_learning)
            echo "predict unspported ${model_name}" > tmp.log
        ;;
        esac

    elif [[ ${input_model_type} == "pretrained" ]];then
        case ${model_type} in
        ImageNet|slim|PULC|DeepHash)
            if [[ -d ${infer_pretrain}_infer ]] && [[ -f ${infer_pretrain}_infer/inference.pdiparams ]];then #有下载好的，或者export已导出的跳过下载
                export pretrained_model="../${infer_pretrain}_infer"
            elif [[ ${model_type} == "PULC" ]];then
                wget -q https://paddleclas.bj.bcebos.com/models/PULC/${infer_pretrain}_infer.tar --no-proxy
            else
                wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/${infer_pretrain}_infer.tar --no-proxy
            fi
        ;;
        GeneralRecognition)
            download_infer_tar picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer
            download_infer_tar general_PPLCNet_x2_5_lite_v1.0_infer
        ;;
        Cartoonface)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar cartoon_rec_ResNet50_iCartoon_v1.0_infer
        ;;
        Logo)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar logo_rec_ResNet50_Logo3K_v1.0_infer
        ;;
        Products)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar product_ResNet50_vd_aliproduct_v1.0_infer
        ;;
        Vehicle)
            download_infer_tar ppyolov2_r50vd_dcn_mainbody_v1.0_infer
            download_infer_tar vehicle_cls_ResNet50_CompCars_v1.0_infer
        ;;
        reid|metric_learning)
            echo "predict unspported ${model_name}" > tmp.log
        ;;
        esac

        if [[ $? -eq 0 ]];then
            if [[ -f ${infer_pretrain}_infer.tar ]];then
                tar xf ${infer_pretrain}_infer.tar
            fi
            export pretrained_model="../${infer_pretrain}_infer"
        else
            echo -e "\033[31m failed! predict pretrained download ${model_name}/${infer_pretrain} failed!\033[0m"
            export pretrained_model="../inference/${model_name}"
        fi
    else
        export pretrained_model="None"
    fi
fi
echo "##### pretrained_model"
echo ${pretrained_model}
