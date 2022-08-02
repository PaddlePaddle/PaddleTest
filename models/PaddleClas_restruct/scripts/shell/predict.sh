# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA 、 trained/pretrained

export yaml_line=${1:-ppcls/configs/ImageNet/ResNet/ResNet50.yaml}
export cuda_type=${2:-SET_MULTI_CUDA}
export input_model_type=${3:-pretrained}

cd ${Project_path} #确定下执行路径
\cp -r -f ${Project_path}/../scripts/shell/prepare.sh .
# #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
source prepare.sh

# arr=("trained" "pretrained") #或者抽象出来到输入参数，现在是默认训好的、预训练的全跑
# for input_model_type in ${arr[@]}
# do
\cp -r -f ${Project_path}/../scripts/shell/choose_model.sh .
export predict_step=True
source choose_model.sh

size_tmp=`cat ${yaml_line} |grep image_shape|cut -d "," -f2|cut -d " " -f2`
#获取train的shape保持和predict一致
cd deploy
sed -i 's/size: 224/size: '${size_tmp}'/g' configs/inference_cls.yaml #修改predict尺寸
sed -i 's/resize_short: 256/resize_short: '${size_tmp}'/g' configs/inference_cls.yaml

echo model_type
echo ${model_type}
case ${model_type} in
ImageNet|slim|metric_learning)
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
Cartoonface)
    python  python/predict_system.py -c configs/inference_cartoon.yaml \
        -o Global.inference_model_dir=${pretrained_model} \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
DeepHash|GeneralRecognition)
    python python/predict_rec.py -c configs/inference_rec.yaml \
        -o Global.rec_inference_model_dir=${pretrained_model} \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
Logo)
    python  python/predict_system.py -c configs/inference_logo.yaml \
        -o Global.inference_model_dir=${pretrained_model} \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
Products)
    python  python/predict_system.py -c configs/inference_product.yaml \
        -o Global.inference_model_dir=${pretrained_model} \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
PULC)
    # 9中方向用 model_type_PULC 区分
    python python/predict_cls.py -c configs/PULC/${model_type_PULC}/inference_/${model_type_PULC}.yaml
        -o Global.inference_model_dir=${pretrained_model} \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
reid)
    echo "predict_exit_code: unspported" > ../${log_path}/predict/${model_name}_${input_model_type}.log
;;
Vehicle)
    python  python/predict_system.py -c configs/inference_vehicle.yaml \
        -o Global.inference_model_dir=${pretrained_model} \
        -o Global.use_gpu=${set_cuda_flag} \
        > ../${log_path}/predict/${model_name}_${input_model_type}.log 2>&1
;;
esac

# if [[ $? -eq 0 ]] \
    # && [[ $(grep -c  "Error" ../${log_path}/predict/${model_name}_${input_model_type}.log) -eq 0 ]];then
if [[ $? -eq 0 ]];then
    echo -e "\033[33m predict of ${model_name}_${input_model_type}  \
    successfully!\033[0m"| tee -a ../${log_path}/result.log
    echo "predict_exit_code: 0.0" >> ../${log_path}/predict/${model_name}_${input_model_type}.log
else
    cat ../${log_path}/predict/${model_name}_${input_model_type}.log
    echo -e "\033[31m predict of ${model_name}_${input_model_type} \
    failed!\033[0m"| tee -a ../${log_path}/result.log
    echo "predict_exit_code: 1.0" >> ../${log_path}/predict/${model_name}_${input_model_type}.log
fi
sed -i 's/size: '${size_tmp}'/size: 224/g' configs/inference_cls.yaml #改回predict尺寸
sed -i 's/resize_short: '${size_tmp}'/resize_short: 256/g' configs/inference_cls.yaml
cd ..
# done
