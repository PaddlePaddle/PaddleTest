# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA 、 trained/pretrained

export yaml_line=${1:-ppcls/configs/ImageNet/ResNet/ResNet50.yaml}
export cuda_type=${2:-SET_MULTI_CUDA}
export input_model_type=${3:-pretrained}
export Project_path=${Project_path:-$PWD}

cd ${Project_path} #确定下执行路径
\cp -r -f ${Project_path}/../scripts/shell/prepare.sh .
# #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
source prepare.sh

\cp -r -f ${Project_path}/../scripts/shell/choose_model.sh .
export predict_step=True

# source choose_model.sh
# 因为训练不足导致预测BN算子报错,直接使用预训练模型  根因是epoch数不能小于5
if [[ ${model_name} == "PULC-language_classification-PPLCNet_x1_0" ]] \
    || [[ ${model_name} == "PULC-language_classification-MobileNetV3_small_x0_35" ]] \
    || [[ ${model_name} == "PULC-textline_orientation-PPLCNet_x1_0" ]];then
    input_model_type_tmp=${input_model_type}
    export input_model_type=pretrained
    source choose_model.sh
    export input_model_type=${input_model_type_tmp}
else
    source choose_model.sh
fi

size_tmp=`cat ${yaml_line} |grep image_shape|cut -d "," -f2|cut -d " " -f2`
#获取train的shape保持和predict一致
cd deploy
sed -i 's/size: 224/size: '${size_tmp}'/g' configs/inference_cls.yaml #修改predict尺寸
sed -i 's/resize_short: 256/resize_short: '${size_tmp}'/g' configs/inference_cls.yaml

echo model_type
echo ${model_type}
case ${model_type} in
ImageNet|slim|DeepHash)
    if [[ ${yaml_line} =~ 'ultra' ]];then
        python python/predict_cls.py -c configs/inference_cls_ch4.yaml  \
            -o Global.infer_imgs="./images"  \
            -o Global.batch_size=4 -o Global.inference_model_dir=${pretrained_model} \
            -o Global.use_gpu=${set_cuda_flag} \
            > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
    else
        python python/predict_cls.py -c configs/inference_cls.yaml  \
            -o Global.infer_imgs="./images"  \
            -o Global.batch_size=4 \
            -o Global.inference_model_dir=${pretrained_model} \
            -o Global.use_gpu=${set_cuda_flag} \
            > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
    fi
;;
GeneralRecognition)
    python  python/predict_system.py -c configs/inference_general.yaml \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
Cartoonface)
    python  python/predict_system.py -c configs/inference_cartoon.yaml \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
Logo)
    python  python/predict_system.py -c configs/inference_logo.yaml \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
Products)
    python  python/predict_system.py -c configs/inference_product.yaml \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
Vehicle)
    python  python/predict_system.py -c configs/inference_vehicle.yaml \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
PULC)
    # 9中方向用 model_type_PULC 区分
    python python/predict_cls.py -c configs/PULC/${model_type_PULC}/inference_${model_type_PULC}.yaml \
        -o Global.inference_model_dir=${pretrained_model} \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
reid|metric_learning)
    echo "predict unspported ${model_name}" > ../${log_path}/predict/${model_name}_${input_model_type}.log
;;
esac

# if [[ $? -eq 0 ]] \
    # && [[ $(grep -c  "Error" ../${log_path}/predict/${model_name}_${input_model_type}.log) -eq 0 ]];then
if [[ $? -eq 0 ]];then
    cat ../${log_path}/predict/${model_name}_${input_model_type}.log
    echo -e "\033[33m successfully! predict of ${model_name}_${input_model_type} successfully!\033[0m" \
        | tee -a ../${log_path}/result.log
    echo "predict_exit_code: 0.0" >> ../${log_path}/predict/${model_name}_${input_model_type}.log
else
    cat ../${log_path}/predict/${model_name}_${input_model_type}.log
    echo -e "\033[31m failed! predict of ${model_name}_${input_model_type} failed!\033[0m" \
        | tee -a ../${log_path}/result.log
    echo "predict_exit_code: 1.0" >> ../${log_path}/predict/${model_name}_${input_model_type}.log
fi
sed -i 's/size: '${size_tmp}'/size: 224/g' configs/inference_cls.yaml #改回predict尺寸
sed -i 's/resize_short: '${size_tmp}'/resize_short: 256/g' configs/inference_cls.yaml
cd ..
