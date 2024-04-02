#!/bin/bash

python -m pip install --upgrade pip
echo -e '*****************paddle_version******'
    python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************detection_version****'
    git rev-parse HEAD

err_sign=false
if [ -d "log" ];then rm -rf log
fi
mkdir log
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err
if [ -d "output" ];then rm -rf output
fi
#cpp infer compile
if [ "${task_type}" == 'release' ];then
cd deploy/cpp
wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xf 3.4.7.tar.gz
mkdir -p opencv-3.4.7/build
cd opencv-3.4.7/build
install_path=/usr/local/opencv3
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path} -DCMAKE_BUILD_TYPE=Release
make -j
make install
cd ..
cd ..
wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.zip
unzip yaml-cpp-0.7.0.zip
mkdir -p yaml-cpp-yaml-cpp-0.7.0/build
cd yaml-cpp-yaml-cpp-0.7.0/build
cmake -DYAML_BUILD_SHARED_LIBS=ON ..
make -j
make install
cd ..
cd ..
wget ${paddle_inference}
tar xvf paddle_inference.tgz
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF
DEMO_NAME=test_seg
work_path=$(dirname $(readlink -f $0))
LIB_DIR="${work_path}/paddle_inference"
mkdir -p build
cd build
rm -rf *
cmake .. \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_GPU=${WITH_GPU} \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_STATIC_LIB=OFF \
  -DPADDLE_LIB=${LIB_DIR}
make -j
cd ..
cd ..
cd ..
fi
# prepare dynamic data
mkdir data
if [ -d "data/cityscapes" ];then rm -rf data/cityscapes
fi
ln -s ${data_path}/cityscape data/cityscapes
if [ -d "data/VOCdevkit" ]; then rm -rf data/VOCdevkit
fi
ln -s ${data_path}/pascalvoc/VOCdevkit data/VOCdevkit
if [ -d "data/ADEChallengeData2016" ]; then rm -rf data/ADEChallengeData2016
fi
ln -s ${data_path}/ADEChallengeData2016 data/ADEChallengeData2016
if [ -d "data/mini_supervisely" ]; then rm -rf data/mini_supervisely
fi
ln -s ${data_path}/mini_supervisely data/mini_supervisely
if [ -d "data/PP-HumanSeg14K" ]; then rm -rf data/PP-HumanSeg14K
fi
ln -s ${data_path}/PP-HumanSeg14K data/PP-HumanSeg14K
if [ -d "data/camvid" ]; then rm -rf data/camvid
fi
ln -s ${data_path}/camvid data/camvid
if [ -d "seg_dynamic_pretrain" ];then rm -rf seg_dynamic_pretrain
fi
ln -s ${data_path}/seg_dynamic_pretrain seg_dynamic_pretrain


print_result(){
    if [ $? -ne 0 ];then
        echo -e "${model},${mode},FAIL"
        echo -e "${model},${mode},Failed" >>result 2>&1
        cd ${log_dir}/log_err
        if [ ! -d ${model} ];then
            mkdir ${model}
        fi
        cd ../${model_type_path}
        mv ${log_dir}/log/${model}/${model}_${mode}.log ${log_dir}/log_err/${model}/
        err_sign=true
        cat ${log_dir}/log_err/${model}/${model}_${mode}.log
        #exit 1
    else
        echo -e "${model},${mode},SUCCESS"
        echo -e "${model},${mode},Passed" >>result 2>&1
    fi
}

# run dynamic models
pip install -r requirements.txt
#install paddleseg
pip install -v -e .
log_dir=.
model_type_path=
if [ "${task_type}" == 'develop_d1' ];then
find . | grep configs | grep .yml | grep -v _base_ | grep -v quick_start | grep -v EISeg | grep -v setr | grep -v portraitnet | grep -v contrib | grep -v Matting | grep -v segformer | grep -v test_tipc | grep -v deeplabv3  | grep -v benchmark | grep -v smrt | grep -v pssl | grep -v mscale_ocrnet | tee dynamic_config_all_temp
elif [ "${task_type}" == 'develop_d2' ];then
find . | grep configs | grep .yml | grep -v _base_ | grep -v quick_start | grep -v setr | grep segformer | grep -v contrib | grep -v EISeg | grep -v Matting | grep -v test_tipc | grep -v benchmark | grep -v smrt | grep -v pssl | tee dynamic_config_segformer
find . | grep configs | grep .yml | grep -v _base_ | grep -v quick_start | grep -v setr | grep deeplabv3 | grep -v contrib | grep -v EISeg | grep -v Matting | grep -v test_tipc | grep -v benchmark | grep -v smrt | grep -v pssl | tee dynamic_config_deeplabv3
cat dynamic_config_segformer dynamic_config_deeplabv3 >>dynamic_config_all_temp
else
find . | grep configs | grep .yml | grep -v _base_ | grep -v quick_start | grep -v EISeg | grep -v setr | grep -v portraitnet | grep -v contrib | grep -v Matting | grep -v segformer | grep -v test_tipc | grep -v benchmark | grep -v smrt | grep -v pssl | grep -v mscale_ocrnet | tee dynamic_config_all_temp
fi
sed -i "s/trainaug/train/g" configs/_base_/pascal_voc12aug.yml
skip_export_model='espnetv1_cityscapes_1024x512_120k enet_cityscapes_1024x512_80k segnet_cityscapes_1024x512_80k gscnn_resnet50_os8_cityscapes_1024x512_80k'
# dynamic fun
TRAIN_MUlTI_DYNAMIC(){
    export CUDA_VISIBLE_DEVICES=$cudaid2
    mode=train_multi_dynamic
    if [[ ${model} =~ 'segformer' || ${model} =~ 'encnet' ]];then
        echo -e "${model} does not test multi_train！"
    else
        python -m paddle.distributed.launch tools/train.py \
           --config ${config} \
           --save_interval 100 \
           --iters 10 \
           --num_workers 8 \
           --batch_size 1 \
           --save_dir output/${model} >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
TRAIN_SINGLE_DYNAMIC(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=train_single_dynamic
    if [[ ${model} =~ 'segformer' || ${model} =~ 'pfpn' || ${model} =~ 'encnet' ]];then
        echo -e "${model} does not test single_train！"
    else
        python tools/train.py \
           --config ${config} \
           --save_interval 100 \
           --iters 10 \
           --batch_size 1 \
           --num_workers 8 \
           --save_dir output/${model} >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
TRAIN_SINGLE_DYNAMIC_BS1(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=train_single_dynamic_bs1
    if [[ ${model} =~ 'segformer' || ${model} =~ 'encnet' ]];then
        echo -e "${model} does not test single_dynamic_bs1_train！"
    else
        python tools/train.py \
           --config ${config} \
           --save_interval 100 \
           --iters 10 \
           --batch_size=1 \
           --num_workers 8 \
           --save_dir output/${model} >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
EVAL_DYNAMIC(){
    export CUDA_VISIBLE_DEVICES=$cudaid2
    mode=eval_dynamic
    python -m paddle.distributed.launch tools/val.py \
       --config ${config} \
       --model_path seg_dynamic_pretrain/${model}/model.pdparams >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
    print_result
}
PREDICT_DYNAMIC(){
    mode=predict_dynamic
    python tools/predict.py \
       --config ${config} \
       --model_path seg_dynamic_pretrain/${model}/model.pdparams \
       --image_path demo/${predict_pic} \
       --save_dir output/${model}/result >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
    print_result
}
EXPORT_DYNAMIC(){
    mode=export_dynamic
    if [[ ${model} =~ 'rtformer' || ${model} =~ 'dmnet' ]];then
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python tools/export.py \
           --config ${config} \
           --model_path seg_dynamic_pretrain/${model}/model.pdparams \
           --save_dir ./inference_model/${model} \
           --input_shape 1 3 1024 2048 >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
        print_result
    elif [[ -z `echo ${skip_export_model} | grep -w ${model}` ]];then
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python tools/export.py \
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
    if [ ! -d ./inference_model/${model} ];then
        echo -e "${model} doesn't run export case,so can't run PYTHON_INFER case!"
    else
        export PYTHONPATH=`pwd`
        python deploy/python/infer.py \
           --config ./inference_model/${model}/deploy.yaml \
           --image_path ./demo/${predict_pic} \
           --save_dir ./python_infer_output/${model} >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
CPP_INFER(){
    if [ "${task_type}" == 'release' ];then
        mode=cpp_infer
        if [ ! -d ./inference_model/${model} ];then
            echo -e "${model} doesn't run export case,so can't run CPP_INFER case!"
        else
            ./deploy/cpp/build/test_seg \
               --model_dir=inference_model/${model} \
               --img_path=demo/${predict_pic} \
               --save_dir=cpp_infer_output/${model} \
               --devices=GPU >${log_dir}/log/${model}/${model}_${mode}.log 2>&1
            print_result
        fi
    else
        echo "${model} is not support c++ predict!"
    fi
}
grep -F -v -f no_upload dynamic_config_all_temp | sort | uniq | tee dynamic_config_all
if [ "$1" ];then
    cat dynamic_config_all | grep "${1}" > config_list_tmp
    mv config_list_tmp dynamic_config_all
fi
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
    wget -P seg_dynamic_pretrain/${model}/ https://paddleseg.bj.bcebos.com/dygraph/pascal_voc12/${model}/model.pdparams
elif [[ -n `echo ${model} | grep cityscapes` ]] && [[ ! -f seg_dynamic_pretrain/${model}/model.pdparams ]];then
    wget -P seg_dynamic_pretrain/${model}/ https://paddleseg.bj.bcebos.com/dygraph/cityscapes/${model}/model.pdparams
elif [[ -n `echo ${model} | grep ade20k` ]] && [[ ! -f seg_dynamic_pretrain/${model}/model.pdparams ]];then
    wget -P seg_dynamic_pretrain/${model}/ https://paddleseg.bj.bcebos.com/dygraph/ade20k/${model}/model.pdparams
elif [[ -n `echo ${model} | grep camvid` ]] && [[ ! -f seg_dynamic_pretrain/${model}/model.pdparams ]];then
    wget -P seg_dynamic_pretrain/${model}/ https://paddleseg.bj.bcebos.com/dygraph/camvid/${model}/model.pdparams
fi
if [ "$2" ];then
    $2
else
if [ ! -s seg_dynamic_pretrain/${model}/model.pdparams ];then
    echo "${model} url is bad!"
else
    TRAIN_MUlTI_DYNAMIC
    TRAIN_SINGLE_DYNAMIC_BS1
    TRAIN_SINGLE_DYNAMIC
    EVAL_DYNAMIC
    PREDICT_DYNAMIC
    EXPORT_DYNAMIC
    PYTHON_INFER_DYNAMIC
    CPP_INFER
fi
fi
done

if [ "${err_sign}" = true ];then
    export status='Failed'
    export exit_code='8'
    exit 1
else
    export status='Passed'
    export exit_code='0'
    exit 0
fi
