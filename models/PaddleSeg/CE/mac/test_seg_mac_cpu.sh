#!/bin/bash


python -m pip install --upgrade pip
echo -e '*****************paddle_version*****'
    python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************detection_version****'
    git rev-parse HEAD
err_sign=falsel
if [ -d "log" ];then rm -rf log
fi
mkdir log
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err
if [ -d "output" ];then rm -rf output
fi


print_result(){
    if [ $? -ne 0 ];then
        echo -e "${model},${mode},FAIL"
        echo -e "${model},${mode},Failed" >result
        cd ${log_dir}/log_err
        if [ ! -d ${model} ];then
            mkdir ${model}
        fi
        cd ../${model_type_path}
        cat ${log_dir}/log/${model}/${model}_${mode}.log
        mv ${log_dir}/log/${model}/${model}_${mode}.log ${log_dir}/log_err/${model}/
        err_sign=true
        #exit 1
    else
        echo -e "${model},${mode},SUCCESS"
        echo -e "${model},${mode},Passed" >result
    fi
}


# run dynamic models
python -m pip install -r requirements.txt
log_dir=.
model_type_path=
sed -i '' "s/trainaug/train/g" configs/_base_/pascal_voc12aug.yml
skip_export_model='gscnn_resnet50_os8_cityscapes_1024x512_80k'
# dynamic fun
TRAIN_SINGLE_DYNAMIC(){
    mode=train_single_dynamic
    python train.py \
       --config ${config} \
       --save_interval 100 \
       --iters 10 \
       --save_dir output/${model} >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
    print_result
}
EVAL_DYNAMIC(){
    mode=eval_dynamic
    python  val.py \
       --config ${config} \
       --model_path seg_dynamic_pretrain/${model}/model.pdparams >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
    print_result
}
PREDICT_DYNAMIC(){
    mode=predict_dynamic
    python predict.py \
       --config ${config} \
       --model_path seg_dynamic_pretrain/${model}/model.pdparams \
       --image_path demo/${predict_pic} \
       --save_dir output/${model}/result >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
    print_result
}
EXPORT_DYNAMIC(){
    mode=export_dynamic
    if [[ -z `echo ${skip_export_model} | grep -w ${model}` ]];then
        python export.py \
           --config ${config} \
           --model_path seg_dynamic_pretrain/${model}/model.pdparams \
           --save_dir ./inference_model/${model} >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
        print_result
    else
        echo -e "${model} does not support export!"
    fi
}
PYTHON_INFER_DYNAMIC(){
    mode=python_infer_dynamic
    if [[ ${model} =~ 'dnlnet' || ${model} =~ 'gscnn' ]];then
        echo -e "${model} does not test python_infer！"
    else
        export PYTHONPATH=`pwd`
        python deploy/python/infer.py \
           --config ./inference_model/${model}/deploy.yaml \
           --image_path ./demo/${predict_pic} \
           --device cpu \
           --save_dir ./python_infer_output/${model} >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
for config in `cat dynamic_config_all`
do
tmp=${config##*/}
model=${tmp%.*}
echo "${model}"
cd log && mkdir ${model}
cd ..
predict_pic='leverkusen_000029_000019_leftImg8bit.png'
if [[ -n `echo ${model} | grep voc12` ]];then
    predict_pic='2007_000033.jpg'
fi
if [[ -n `echo ${model} | grep voc12` ]] && [[ ! -f seg_dynamic_pretrain/${model}/model.pdparams ]];then
    wget -P seg_dynamic_pretrain/${model}/ https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/${model}/model.pdparams
    if [ ! -s seg_dynamic_pretrain/${model}/model.pdparams ];then
        echo "${model} url is bad!"
    else
        TRAIN_SINGLE_DYNAMIC
        EVAL_DYNAMIC
        PREDICT_DYNAMIC
        EXPORT_DYNAMIC
        PYTHON_INFER_DYNAMIC
    fi
elif [[ -z `echo ${model} | grep voc12` ]] && [[ ! -f seg_dynamic_pretrain/${model}/model.pdparams ]];then
    wget -P seg_dynamic_pretrain/${model}/ https://bj.bcebos.com/paddleseg/dygraph/cityscapes/${model}/model.pdparams
    if [ ! -s seg_dynamic_pretrain/${model}/model.pdparams ];then
        echo "${model} url is bad!"
    else
        TRAIN_SINGLE_DYNAMIC
        EVAL_DYNAMIC
        PREDICT_DYNAMIC
        EXPORT_DYNAMIC
        PYTHON_INFER_DYNAMIC
    fi
else
    TRAIN_SINGLE_DYNAMIC
    EVAL_DYNAMIC
    PREDICT_DYNAMIC
    EXPORT_DYNAMIC
    PYTHON_INFER_DYNAMIC
fi
done

if [ "${err_sign}" = true ];then
    exit 1
else
    exit 0
fi
