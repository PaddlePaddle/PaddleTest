#!/bin/bash


python -m pip install --upgrade pip
python -m pip install Cython --ignore-installed;
python -m pip install -r requirements.txt --ignore-installed;
python -m pip install cython_bbox --ignore-installed;
#brew update
brew install guile
brew install libidn
brew install ffmpeg

function i_sed()
{
    if [[ -e /usr/local/bin/gsed ]];then
        gsed -i "$@"
    else
        sed -i '' "$@"
    fi
}

echo -e '*****************paddle_version*****'
python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************detection_version****'
git rev-parse HEAD
#create log dir
if [ -d "log" ];then rm -rf log
fi
mkdir log
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err

# prepare dynamic data
i_sed "s/trainval.txt/test.txt/g" configs/datasets/voc.yml
# modify dynamic_train_iter
i_sed 's/for step_id, data in enumerate(self.loader):/for step_id, data in enumerate(self.loader):\n                if step_id == 10: break/g' ppdet/engine/trainer.py
i_sed 's/for seq in seqs/for seq in [seqs[0]]/g' ppdet/engine/tracker.py
i_sed 's/for step_id, data in enumerate(dataloader):/for step_id, data in enumerate(dataloader):\n            if step_id == 10: break/g' ppdet/engine/tracker.py
#modify coco images
i_sed 's/coco.getImgIds()/coco.getImgIds()[:2]/g' ppdet/data/source/coco.py
i_sed 's/coco.getImgIds()/coco.getImgIds()[:2]/g' ppdet/data/source/keypoint_coco.py
i_sed 's/records, cname2cid/records[:2], cname2cid/g' ppdet/data/source/voc.py

print_result(){
    if [ $? -ne 0 ];then
        echo -e "${model},${model_type},${mode},FAIL"
        cd log_err
        if [ ! -d ${model} ];then
            mkdir ${model}
        fi
        cd ../
        cat log/${model}/${model}_${model_type}_${mode}.log
        mv log/${model}/${model}_${model_type}_${mode}.log log_err/${model}/
        err_sign=true
    else
        echo -e "${model},${model_type},${mode},SUCCESS"
    fi
}

TRAIN_CPU(){
    mode=train_cpu
    export CPU_NUM=10
    python tools/train.py \
           -c ${config} \
           -o TrainReader.batch_size=1 epoch=1 use_gpu=false >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
TRAIN_WITH_EVAL(){
    mode=train_with_eval
    export CPU_NUM=10
    python tools/train.py \
           -c ${config} \
           -o TrainReader.batch_size=1 epoch=1 use_gpu=false \
           --eval >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EVAL(){
    mode=eval
    python tools/eval.py \
           -c ${config} \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams EvalReader.batch_size=1 use_gpu=false >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}

EVAL_MOT(){
    mode=eval
    python tools/eval_mot.py \
           -c ${config} \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams use_gpu=false >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}

INFER(){
    mode=infer
    python tools/infer.py \
           -c ${config} \
           --infer_img=${image} \
           --output_dir=infer_output/${model}/ \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams use_gpu=false >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
INFER_MOT(){
    mode=infer
    python tools/infer_mot.py \
           -c ${config} \
           --video_file=test_demo.mp4 \
           --output_dir=infer_output/${model}/ \
           --save_videos \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams use_gpu=false >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EXPORT(){
    mode=export
    python tools/export_model.py \
           -c ${config} \
           --output_dir=inference_model \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams use_gpu=false >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
PYTHON_INFER(){
    mode=python_infer
    python deploy/python/infer.py \
           --model_dir=inference_model/${model} \
           --image_file=${image} \
           --run_mode=paddle \
           --device=CPU \
           --threshold=0.5 \
           --output_dir=python_infer_output/${model}/ >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
MOT_PYTHON_INFER(){
    mode=mot_python_infer
    export PYTHONPATH=`pwd`
    python deploy/python/mot_jde_infer.py \
           --model_dir=./inference_model/${model} \
           --video_file=test_demo.mp4 \
           --device=CPU \
           --save_mot_txts \
           --output_dir=python_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
POSE_PYTHON_INFER(){
    mode=pose_python_infer
    python deploy/python/keypoint_infer.py \
           --model_dir=./inference_model/${model} \
           --image_file=demo/hrnet_demo.jpg \
           --device=CPU \
           --output_dir=python_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}

model_list='ppyolov2_r50vd_dcn_365e_coco yolov3_darknet53_270e_coco solov2_r50_fpn_1x_coco faster_rcnn_r50_fpn_1x_coco mask_rcnn_r50_1x_coco cascade_rcnn_r50_fpn_1x_coco ssd_mobilenet_v1_300_120e_voc ttfnet_darknet53_1x_coco fcos_r50_fpn_1x_coco hrnet_w32_256x192 fairmot_dla34_30e_1088x608'
model_s2anet='s2anet_conv_2x_dota'
model_mot='fairmot_dla34_30e_1088x608'
model_keypoint='hrnet_w32_256x192'
model_skip_train_eval='ppyolov2_r50vd_dcn_365e_coco fairmot_dla34_30e_1088x608'
err_sign=false
model_type=dynamic
for model in ${model_list}
do
weight_dir=
if [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    weight_dir=keypoint/
elif [[ -n `echo "${model_mot}" | grep -w "${model}"` ]];then
    weight_dir=mot/
fi
image=demo/000000570688.jpg
if [[ -n `echo "${model_s2anet}" | grep -w "${model}"` ]];then
    image=demo/P0072__1.0__0___0.png
elif [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    image=demo/hrnet_demo.jpg
fi
config=`cat model_list_ci | grep ${model}`
cd log && mkdir ${model} && cd ..
TRAIN_CPU
if [[ -n `echo "${model_skip_train_eval}" | grep -w "${model}"` ]];then
    echo -e "skip train with eval for model ${model}!"
else
    TRAIN_WITH_EVAL
fi
if [[ -n `echo "${model_mot}" | grep -w "${model}"` ]];then
    EVAL_MOT
    INFER_MOT
else
    EVAL
    INFER
fi
EXPORT
if [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    POSE_PYTHON_INFER
elif [[ -n `echo "${model_mot}" | grep -w "${model}"` ]];then
    MOT_PYTHON_INFER
else
    PYTHON_INFER
fi
done
if [ "${err_sign}" = true ];then
    exit 1
else
    exit 0
fi
