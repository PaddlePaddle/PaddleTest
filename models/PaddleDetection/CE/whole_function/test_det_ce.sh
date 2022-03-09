#!/bin/bash

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

echo "$1"
if [ "$1" != 'release' ];then
python -m pip install --upgrade pip --ignore-installed;
pip install Cython --ignore-installed;
pip install -r requirements.txt --ignore-installed;
pip install cython_bbox --ignore-installed;
rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
yum install ffmpeg ffmpeg-devel -y
else
python -m pip install --upgrade pip --ignore-installed;
pip install Cython --ignore-installed;
pip install -r requirements.txt --ignore-installed;
pip install cython_bbox --ignore-installed;
apt-get update && apt-get install -y ffmpeg
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
sed -i "s|TENSORRT_LIB_DIR=/path/to/tensorrt/lib|TENSORRT_LIB_DIR=/usr/local/TensorRT-6.0.1.8/lib|g" scripts/build.sh
sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/x86_64-linux-gnu|g" scripts/build.sh
sh scripts/build.sh
cd ../..
fi
if [ -d 'dataset/coco' ];then
rm -rf dataset/coco
fi
ln -s ${data_path}/data/coco dataset/coco
if [ -d 'dataset/voc' ];then
rm -rf dataset/voc
fi
ln -s ${data_path}/../PaddleSeg/pascalvoc dataset/voc
if [ -d "dataset/mot" ];then rm -rf dataset/mot
fi
ln -s ${data_path}/data/mot dataset/mot
if [ -d "dataset/mpii" ];then rm -rf dataset/mpii
fi
ln -s ${data_path}/data/mpii_tar dataset/mpii
if [ -d "dataset/DOTA_1024_s2anet" ];then rm -rf dataset/DOTA_1024_s2anet
fi
ln -s ${data_path}/data/DOTA_1024_s2anet dataset/DOTA_1024_s2anet
if [ -d "dataset/VisDrone2019_coco" ];then rm -rf dataset/VisDrone2019_coco
fi
ln -s ${data_path}/data/VisDrone2019_coco dataset/VisDrone2019_coco
if [ -d "dataset/mainbody" ];then rm -rf dataset/mainbody
fi
ln -s ${data_path}/data/mainbody dataset/mainbody
if [ -d "dataset/aic_coco_train_cocoformat.json" ];then rm -f dataset/aic_coco_train_cocoformat.json
fi
ln -s ${data_path}/data/aic_coco_train_cocoformat.json dataset/aic_coco_train_cocoformat.json
if [ -d "dataset/AIchallenge" ];then rm -rf dataset/AIchallenge
fi
ln -s ${data_path}/data/AIchallenge dataset/AIchallenge
if [ -d "dataset/spine_coco" ];then rm -rf dataset/spine_coco
fi
ln -s ${data_path}/data/spine_coco dataset/spine_coco
if [ -d "/root/.cache/paddle/weights" ];then rm -rf /root/.cache/paddle/weights
fi
ln -s ${data_path}/data/ppdet_pretrained /root/.cache/paddle/weights

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

if [ "$1" == 'develop_d1' ];then
find . | grep .yml | grep -v benchmark |  grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep -v mot | grep -v cascade_rcnn | grep -v centernet | grep -v picodet | grep -v yolov3 | grep -v ssd | grep -v dcn | grep -v faster_rcnn  | grep -v mask_rcnn | awk '{print $NF}' | tee config_list
elif [ "$1" == 'develop_d2' ];then
find . | grep .yml | grep -v benchmark |  grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep -v mot | grep -v cascade_rcnn | grep -v centernet | grep -v picodet | grep yolov3 | awk '{print $NF}' | tee config_list1
find . | grep .yml | grep -v benchmark |  grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep -v mot | grep -v cascade_rcnn | grep -v centernet | grep -v picodet | grep faster_rcnn | awk '{print $NF}' | tee config_list2
cat config_list1 config_list2 >>config_list
elif [ "$1" == 'develop_d3' ];then
find . | grep .yml | grep -v benchmark | grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep mot | awk '{print $NF}' | tee config_list3
find . | grep .yml | grep -v benchmark |  grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep -v mot | grep -v cascade_rcnn | grep -v centernet | grep -v picodet | grep ssd | awk '{print $NF}' | tee config_list4
find . | grep .yml | grep -v benchmark |  grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep -v mot | grep -v cascade_rcnn | grep -v centernet | grep -v picodet | grep -v yolov3 | grep -v ssd | grep -v dcn | grep -v faster_rcnn  | grep mask_rcnn | awk '{print $NF}' | tee mask_list
cat  mask_list config_list3 config_list4 >>config_list
elif [ "$1" == 'develop_d4' ];then
find . | grep .yml | grep -v benchmark | grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep cascade_rcnn | awk '{print $NF}' | tee cascade_list
find . | grep .yml | grep -v benchmark | grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep centernet | awk '{print $NF}' | tee centernet_list
find . | grep .yml | grep -v benchmark | grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep picodet | awk '{print $NF}' | tee picodet_list
find . | grep .yml | grep -v benchmark |  grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v deepsort | grep -v test | grep  -v minicoco | grep -v mot | grep -v cascade_rcnn | grep -v centernet | grep -v picodet | grep dcn | awk '{print $NF}' | tee config_list5
cat cascade_list centernet_list picodet_list config_list5 >>config_list
else
find . | grep .yml | grep -v benchmark | grep configs | grep -v static | grep -v _base_ | grep -v datasets | grep -v runtime | grep -v slim | grep -v roadsign | grep -v test  | grep  -v minicoco | grep -v deepsort | grep -v gfl | awk '{print $NF}' | tee config_list
fi

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
            export PYTHONPATH=`pwd`
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
        export PYTHONPATH=`pwd`
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
            if [[ -n `echo "${model_skip_export}" | grep -w "${model}"` ]];then
                echo "${model} does not support EXPORT!"
            else
                mode=export
                python tools/export_model.py \
                       -c ${config} \
                       --output_dir=inference_model \
                       -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
                print_result
            fi
        fi
    else
        if [[ -n `echo "${model_skip_export}" | grep -w "${model}"` ]];then
            echo "${model} doesn't support EXPORT!"
        else
            mode=export
            python tools/export_model.py \
                   -c ${config} \
                   --output_dir=inference_model \
                   -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    fi
}
PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run PYTHON_INFER case!"
    else
        if [[ -n `echo "${model_skip_pyinfer}" | grep -w "${model}"` || -n `echo "${model_skip_export}" | grep -w "${model}"` ]] ;then
            echo "${model} doesn't support PYTHON_INFER case!"
        else
            mode=python_infer
            export CUDA_VISIBLE_DEVICES=$cudaid1
            export PYTHONPATH=`pwd`
            python deploy/python/infer.py \
                   --model_dir=inference_model/${model} \
                   --image_file=${image} \
                   --run_mode=paddle \
                   --device=GPU \
                   --threshold=0.5 \
                   --output_dir=python_infer_output/${model}/ >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    fi
}
MOT_PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run MOT_PYTHON_INFER case!"
    else
        if [[ -n `echo "${model_skip_pyinfer}" | grep -w "${model}"` || -n `echo "${model_skip_export}" | grep -w "${model}"` ]] ;then
            echo "${model} doesn't support MOT_PYTHON_INFER case!"
        else
            mode=mot_python_infer
            export CUDA_VISIBLE_DEVICES=$cudaid1
            export PYTHONPATH=`pwd`
            python deploy/python/mot_jde_infer.py \
                   --model_dir=./inference_model/${model} \
                   --video_file=test_demo.mp4 \
                   --device=GPU \
                   --save_mot_txts \
                   --output_dir=python_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
            print_result
        fi
    fi
}
POSE_PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run POSE_PYTHON_INFER case!"
    else
        if [[ -n `echo "${model_skip_pyinfer}" | grep -w "${model}"` || -n `echo "${model_skip_export}" | grep -w "${model}"` ]] ;then
            echo "${model} doesn't support MOT_PYTHON_INFER case!"
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
    fi
}
CPP_INFER(){
    if [ "$1" == 'release' ];then
        if [ ! -d ./inference_model/${model} ];then
            echo "${model} doesn't run export case,so can't run CPP_INFER case!"
        else
            if [[ -z `echo "${model_skip_cpp}" | grep -w "${model}"` ]];then
                mode=cpp_infer
                ./deploy/cpp/build/main \
                    --model_dir=inference_model/${model} \
                    --image_file=${image} \
                    --device=GPU \
                    --run_mode=paddle \
                    --threshold=0.5 \
                    --output_dir=cpp_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
                print_result
            else
                echo "${model} is not support c++ predict!"
            fi
        fi
    else
        echo "centos does not support cpp infer"
    fi
}
find . | grep configs | grep yml | grep mot | grep -v datasets | grep -v _base_ | grep -v deepsort >model_mot
find . | grep configs | grep yml | grep keypoint | grep -v datasets | grep -v _base_ | grep -v static >model_keypoint
model_s2anet='s2anet_conv_1x_dota s2anet_1x_dota s2anet_1x_spine s2anet_alignconv_2x_dota s2anet_conv_2x_dota'
model_skip_export='fairmot_enhance_hardnet85_30e_1088x608 tood_r50_fpn_1x_coco sparse_rcnn_r50_fpn_3x_pro100_coco sparse_rcnn_r50_fpn_3x_pro300_coco detr_r50_1x_coco gflv2_r50_fpn_1x_coco deformable_detr_r50_1x_coco faster_rcnn_swin_tiny_fpn_1x_coco faster_rcnn_swin_tiny_fpn_2x_coco faster_rcnn_swin_tiny_fpn_3x_coco'
model_skip_pyinfer='retinanet_r50_fpn_mstrain_1x_coco retinanet_r50_fpn_1x_coco picodet_s_320_pedestrian'
model_skip_cpp='centernet_mbv3_small_140e_coco centernet_mbv3_large_140e_coco centernet_shufflenetv2_140e_coco centernet_mbv1_140e_coco centernet_shufflenetv2_1x_140e_coco centernet_mbv3_small_1x_140e_coco centernet_mbv3_large_1x_140e_coco centernet_mbv1_1x_140e_coco centernet_r50_140e_coco centernet_dla34_140e_coco solov2_r50_fpn_1x_coco solov2_r50_fpn_3x_coco solov2_r101_vd_fpn_3x_coco solov2_r50_enhance_coco faster_rcnn_r101_fpn_1x_coco ppyolov2_r50vd_dcn_voc yolov3_darknet53_270e_voc yolov3_darknet53_original_270e_coco tood_r50_fpn_1x_coco'
err_sign=false
model_type=dynamic
for config in `cat config_list`
do
tmp=${config##*/}
model=${tmp%.*}
weight_dir=
if [[ -n `cat model_keypoint | grep -w "${model}"` ]];then
    weight_dir=keypoint/
elif
   [[ -n `cat model_mot | grep -w "${model}"` ]];then
    weight_dir=mot/
fi
image=demo/000000570688.jpg
if [[ -n `echo "${model_s2anet}" | grep -w "${model}"` ]];then
    image=demo/P0072__1.0__0___0.png
elif [[ -n `cat model_keypoint | grep -w "${model}"` ]];then
    image=demo/hrnet_demo.jpg
fi
cd log && mkdir ${model} && cd ..
if [[ -n `echo "${model}" | grep "pedestrian_yolov3_darknet"` ]];then
    image=configs/pedestrian/demo/001.png
    INFER
    EXPORT
    PYTHON_INFER
    CPP_INFER "$1"
elif [[ -n `echo "${model}" | grep "vehicle_yolov3_darknet"` ]];then
    image=configs/vehicle/demo/003.png
    INFER
    EXPORT
    PYTHON_INFER
    CPP_INFER "$1"
elif [[ -n `cat model_mot | grep -w "${model}"` ]];then
    TRAIN
    EVAL_MOT
    INFER_MOT
    EXPORT
    MOT_PYTHON_INFER
elif [[ -n `cat model_keypoint | grep -w "${model}"` ]];then
    TRAIN
    EVAL
    INFER
    EXPORT
    POSE_PYTHON_INFER
else
    TRAIN
    EVAL
    INFER
    EXPORT
    PYTHON_INFER
    CPP_INFER "$1"
fi
done
if [ "${err_sign}" == true ];then
    exit 1
else
    exit 0
fi
