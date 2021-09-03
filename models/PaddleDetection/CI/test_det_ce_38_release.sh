#!/bin/bash

echo -e '*****************paddle_version*****'
python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************detection_version****'
git rev-parse HEAD

python -m pip install pip==20.2.4 --ignore-installed;
pip install Cython --ignore-installed;
pip install -r requirements.txt --ignore-installed;
pip install cython_bbox --ignore-installed;
apt-get update && apt-get install -y ffmpeg

#create log dir
if [ -d "log" ];then rm -rf log
fi
mkdir log
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err
#compile cpp_infer
cd deploy/cpp
# dynamic c++ compile
if [ -f "${paddle_inference}" ];then rm -rf ${paddle_inference}
fi
unset http_proxy
unset https_proxy
wget ${paddle_inference}
export http_proxy=${http_proxy}
export https_proxy=${http_proxy}
tar xvf paddle_inference.tgz
sed -i "s|WITH_GPU=OFF|WITH_GPU=ON|g" scripts/build.sh
sed -i "s|WITH_TENSORRT=OFF|WITH_TENSORRT=ON|g" scripts/build.sh
sed -i "s|CUDA_LIB=/path/to/cuda/lib|CUDA_LIB=/usr/local/cuda/lib64|g" scripts/build.sh
sed -i "s|/path/to/paddle_inference|../paddle_inference_install_dir|g" scripts/build.sh
sed -i "s/TENSORRT_LIB_DIR=\/path\/to\/tensorrt\/lib/TENSORRT_LIB_DIR=\/usr\/local\/TensorRT-6.0.1.8\/lib/g" scripts/build.sh
sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/x86_64-linux-gnu|g" scripts/build.sh
sh scripts/build.sh
cd ../..
#compile op
cd ppdet/ext_op
python setup.py install
cd ../..
# prepare dynamic data
sed -i "s/trainval.txt/test.txt/g" configs/datasets/voc.yml
#modify coco images
sed -i 's/coco.getImgIds()/coco.getImgIds()[:2]/g' ppdet/data/source/coco.py
sed -i 's/coco.getImgIds()/coco.getImgIds()[:2]/g' ppdet/data/source/keypoint_coco.py
sed -i 's/records, cname2cid/records[:2], cname2cid/g' ppdet/data/source/voc.py
# modify dynamic_train_iter
sed -i '/for step_id, data in enumerate(self.loader):/i\            max_step_id =1' ppdet/engine/trainer.py
sed -i '/for step_id, data in enumerate(self.loader):/a\                if step_id == max_step_id: break' ppdet/engine/trainer.py
#modify eval iter
#sed -i '/for step_id, data in enumerate(loader):/i\        max_step_id =1' ppdet/engine/trainer.py
#sed -i '/for step_id, data in enumerate(loader):/a\            if step_id == max_step_id: break' ppdet/engine/trainer.py
#modify mot_eval iter
sed -i '/for seq in seqs/for seq in [seqs[0]]/g' ppdet/engine/tracker.py
sed -i '/for step_id, data in enumerate(dataloader):/i\        max_step_id=1' ppdet/engine/tracker.py
sed -i '/for step_id, data in enumerate(dataloader):/a\            if step_id == max_step_id: break' ppdet/engine/tracker.py
if [ -d 'dataset/coco' ];then
rm -rf dataset/coco
fi
ln -s ${data_path}/coco dataset/coco
if [ -d 'dataset/voc' ];then
rm -rf dataset/voc
fi
ln -s ${data_path}/../PaddleSeg/pascalvoc dataset/voc
if [ -d "dataset/mot" ];then rm -rf dataset/mot
fi
ln -s ${data_path}/mot dataset/mot
if [ -d "dataset/mpii" ];then rm -rf dataset/mpii
fi
ln -s ${data_path}/mpii_tar dataset/mpii
if [ -d "dataset/DOTA_1024_s2anet" ];then rm -rf dataset/DOTA_1024_s2anet
fi
ln -s ${data_path}/DOTA_1024_s2anet dataset/DOTA_1024_s2anet
find . | grep .yml | grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v test  | grep  -v minicoco | grep -v deepsort | grep -v picodet | grep -v gfl | awk '{print $NF}' | tee config_list
#find . | grep .yml | grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v test  | grep  -v minicoco | awk '{print $NF}' | tee config_list


print_result(){
    if [ $? -ne 0 ];then
        echo -e "${model},${model_type},${mode},FAIL"
        cd log_err
        if [ ! -d ${model} ];then
            mkdir ${model}
        fi
        cd ../${model_path}
        cat log/${model}/${model}_${model_type}_${mode}.log
        mv log/${model}/${model}_${model_type}_${mode}.log log_err/${model}/
        err_sign=true
    else
        echo -e "${model},${model_type},${mode},SUCCESS"
    fi
}
TRAIN(){
    export CUDA_VISIBLE_DEVICES=$cudaid2
    mode=train
    python -m paddle.distributed.launch \
    tools/train.py \
           -c ${config} \
           -o TrainReader.batch_size=1 epoch=1 >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EVAL(){
    if [ ! -f /root/.cache/paddle/weights/${model}.pdparams ];then
        wget -P /root/.cache/paddle/weights/ https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams
        if [ ! -s /root/.cache/paddle/weights/${model}.pdparams ];then
            echo "${model} url is bad,so can't run EVAL!"
        else
            export CUDA_VISIBLE_DEVICES=$cudaid1
            mode=eval
            python tools/eval.py \
                   -c ${config} \
                   -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    else
        export CUDA_VISIBLE_DEVICES=$cudaid1
        mode=eval
        python tools/eval.py \
              -c ${config} \
              -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
EVAL_MOT(){
    if [ ! -f /root/.cache/paddle/weights/${model}.pdparams ];then
        wget -P /root/.cache/paddle/weights/ https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams
        if [ ! -s /root/.cache/paddle/weights/${model}.pdparams ];then
            echo "${model} url is bad,so can't run EVAL_MOT!"
        else
            export CUDA_VISIBLE_DEVICES=$cudaid1
            mode=eval
            python tools/eval_mot.py \
                   -c ${config} \
                   -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    else
        export CUDA_VISIBLE_DEVICES=$cudaid1
        mode=eval
        python tools/eval_mot.py \
               -c ${config} \
               -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
INFER(){
    if [ ! -f /root/.cache/paddle/weights/${model}.pdparams ];then
        wget -P /root/.cache/paddle/weights/ https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams
        if [ ! -s /root/.cache/paddle/weights/${model}.pdparams ];then
            echo "${model} url is bad,so can't run INFER!"
        else
            export CUDA_VISIBLE_DEVICES=$cudaid1
            mode=infer
            python tools/infer.py \
                   -c ${config} \
                   --infer_img=${image} \
                   --output_dir=infer_output/${model}/ \
                   -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    else
        export CUDA_VISIBLE_DEVICES=$cudaid1
        mode=infer
        python tools/infer.py \
               -c ${config} \
               --infer_img=${image} \
               --output_dir=infer_output/${model}/ \
               -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
INFER_MOT(){
    if [ ! -f /root/.cache/paddle/weights/${model}.pdparams ];then
        wget -P /root/.cache/paddle/weights/ https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams
        if [ ! -s /root/.cache/paddle/weights/${model}.pdparams ];then
            echo "${model} url is bad,so can't run INFER_MOT!"
        else
            export CUDA_VISIBLE_DEVICES=$cudaid1
            mode=infer
            python tools/infer_mot.py \
                   -c ${config} \
                   --video_file=test_demo.mp4 \
                   --output_dir=infer_output/${model}/ \
                   --save_videos \
                   -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    else
        export CUDA_VISIBLE_DEVICES=$cudaid1
        mode=infer
        python tools/infer_mot.py \
               -c ${config} \
               --video_file=test_demo.mp4 \
               --output_dir=infer_output/${model}/ \
               --save_videos \
               -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
EXPORT(){
    if [ ! -f /root/.cache/paddle/weights/${model}.pdparams ];then
        wget -P /root/.cache/paddle/weights/ https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams
        if [ ! -s /root/.cache/paddle/weights/${model}.pdparams ];then
            echo "${model} url is bad,so can't run EXPORT!"
        else
            mode=export
            python tools/export_model.py \
                   -c ${config} \
                   --output_dir=inference_model \
                   -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    else
        mode=export
        python tools/export_model.py \
               -c ${config} \
               --output_dir=inference_model \
               -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run PYTHON_INFER case!"
    else
        mode=python_infer
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python deploy/python/infer.py \
               --model_dir=inference_model/${model} \
               --image_file=${image} \
               --run_mode=fluid \
               --device=GPU \
               --threshold=0.5 \
               --output_dir=python_infer_output/${model}/ >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
MOT_PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run MOT_PYTHON_INFER case!"
    else
        mode=mot_python_infer
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python deploy/python/mot_jde_infer.py \
               --model_dir=./inference_model/${model} \
               --video_file=test_demo.mp4 \
               --device=GPU \
               --save_mot_txts \
               --output_dir=python_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
POSE_PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run POSE_PYTHON_INFER case!"
    else
        mode=pose_python_infer
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python deploy/python/keypoint_infer.py \
               --model_dir=./inference_model/${model} \
               --image_file=demo/hrnet_demo.jpg \
               --device=GPU \
               --output_dir=python_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    fi
}
CPP_INFER(){
    if [[ -z `echo "${model_skip_cpp}" | grep -w "${model}"` ]];then
        mode=cpp_infer
        ./deploy/cpp/build/main \
            --model_dir=inference_model/${model} \
            --image_file=${image} \
            --device=GPU \
            --run_mode=fluid \
            --threshold=0.5 \
            --output_dir=cpp_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
        print_result
    else
        echo "${model} is not support c++ predict!"
    fi
}
model_s2anet='s2anet_conv_1x_dota s2anet_1x_dota s2anet_1x_spine s2anet_alignconv_2x_dota s2anet_conv_2x_dota'
model_mot='fairmot_dla34_30e_1088x608 fairmot_dla34_30e_1088x608_headtracking21 fairmot_dla34_30e_1088x608_kitticars deepsort_pcb_pyramid_r101 deepsort_yolov3_pcb_pyramid_r101 jde_darknet53_30e_1088x608 jde_darknet53_30e_576x320 jde_darknet53_30e_864x480'
model_keypoint='hrnet_w32_256x192 hrnet_w32_256x256_mpii hrnet_w32_384x288 dark_hrnet_w32_256x192 dark_hrnet_w32_384x288 dark_hrnet_w48_256x192 higherhrnet_hrnet_w32_512 higherhrnet_hrnet_w32_512_swahr higherhrnet_hrnet_w32_640'
model_skip_export='sparse_rcnn_r50_fpn_3x_pro100_coco sparse_rcnn_r50_fpn_3x_pro300_coco detr_r50_1x_coco'
model_skip_cpp='solov2_r50_fpn_1x_coco solov2_r50_fpn_3x_coco solov2_r101_vd_fpn_3x_coco solov2_r50_enhance_coco'

err_sign=false
model_type=dynamic
for config in `cat config_list`
do
tmp=${config##*/}
model=${tmp%.*}
weight_dir=
if [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    weight_dir=keypoint/
elif
   [[ -n `echo "${model_mot}" | grep -w "${model}"` ]];then
    weight_dir=mot/
fi
image=demo/000000570688.jpg
if [[ -n `echo "${model_s2anet}" | grep -w "${model}"` ]];then
    image=demo/P0072__1.0__0___0.png
elif [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    image=demo/hrnet_demo.jpg
fi
cd log && mkdir ${model} && cd ..
if [[ -n `echo "${model}" | grep "pedestrian"` ]];then
    image=configs/pedestrian/demo/001.png
    INFER
    EXPORT
    PYTHON_INFER
    CPP_INFER
elif [[ -n `echo "${model}" | grep "vehicle"` ]];then
    image=configs/vehicle/demo/003.png
    INFER
    EXPORT
    PYTHON_INFER
    CPP_INFER
elif [[ -n `echo "${model_mot}" | grep -w "${model}"` ]];then
    TRAIN
    EVAL_MOT
    INFER_MOT
    EXPORT
    MOT_PYTHON_INFER
elif [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    TRAIN
    EVAL
    INFER
    EXPORT
    POSE_PYTHON_INFER
elif [[ -n `echo "${model_skip_export}" | grep -w "${model}"` ]];then
    TRAIN
    EVAL
    INFER
else
    TRAIN
    EVAL
    INFER
    EXPORT
    PYTHON_INFER
    CPP_INFER
fi
done
if [ "${err_sign}" = true ];then
    exit 1
fi
