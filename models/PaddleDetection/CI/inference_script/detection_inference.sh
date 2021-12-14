set -x
if [ -d "logs" ];then rm -rf logs
fi
mkdir logs
if [ -d "logs_cpp" ];then rm -rf logs_cpp
fi
mkdir logs_cpp
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err
cd log_err
if [ -d "cpp_infer" ];then rm -rf cpp_infer
fi
mkdir cpp_infer
if [ -d "python_infer" ];then rm -rf python_infer
fi
mkdir python_infer
cd ..
#machine type
export no_proxy=bcebos.com;
MACHINE_TYPE=`uname -m`
echo "MACHINE_TYPE: "${MACHINE_TYPE}
config_list='ppyolo_r50vd_dcn_1x_coco ppyolov2_r50vd_dcn_365e_coco yolov3_darknet53_270e_coco solov2_r50_fpn_1x_coco faster_rcnn_r50_fpn_1x_coco mask_rcnn_r50_1x_coco s2anet_conv_2x_dota ssd_mobilenet_v1_300_120e_voc ttfnet_darknet53_1x_coco fcos_r50_fpn_1x_coco'
config_list_cpp='ppyolo_r50vd_dcn_1x_coco ppyolov2_r50vd_dcn_365e_coco yolov3_darknet53_270e_coco faster_rcnn_r50_fpn_1x_coco mask_rcnn_r50_1x_coco s2anet_conv_2x_dota ssd_mobilenet_v1_300_120e_voc ttfnet_darknet53_1x_coco fcos_r50_fpn_1x_coco'
config_skip_trt8='ppyolo_r50vd_dcn_1x_coco ppyolov2_r50vd_dcn_365e_coco solov2_r50_fpn_1x_coco faster_rcnn_r50_fpn_1x_coco mask_rcnn_r50_1x_coco ttfnet_darknet53_1x_coco fcos_r50_fpn_1x_coco s2anet_conv_2x_dota'
config_skip_bs2='solov2_r50_fpn_1x_coco mask_rcnn_r50_1x_coco s2anet_conv_2x_dota'
config_skip_video='mask_rcnn_r50_1x_coco'
config_s2anet='s2anet_conv_2x_dota'
mode_list='trt_fp32 trt_fp16 trt_int8 paddle'
err_sign=false
print_result_python(){
    if [ $? -ne 0 ];then
        echo -e "${config}_${mode},python_infer,FAIL"
        cd log_err/python_infer
        if [ ! -d ${config} ];then
            mkdir ${config}
        fi
        cd ../..
        mv logs/${config}_${mode}.log log_err/python_infer/${config}/
        err_sign=true
    else
        echo -e "${config}_${mode},python_infer,SUCCESS"
    fi
}
python_trt(){
    python deploy/python/infer.py \
           --model_dir=./inference_model/${config} \
           --image_file=${image} \
           --device=GPU \
           --run_mode=${mode} \
           --threshold=0.5 \
           --trt_calib_mode=${trt_calib_mode} \
           --output_dir=python_infer_output/${config}_${mode} >logs/${config}_${mode}.log 2>&1
    print_result_python
}
python_cpu(){
    mode=cpu
    python deploy/python/infer.py \
           --model_dir=./inference_model/${config} \
           --image_file=${image} \
           --device=CPU \
           --threshold=0.5 \
           --output_dir=python_infer_output/${config}_${mode} >logs/${config}_${mode}.log 2>&1
    print_result_python
}
python_mkldnn(){
    mode=mkldnn
    python deploy/python/infer.py \
           --model_dir=./inference_model/${config} \
           --image_file=${image} \
           --device=CPU \
           --threshold=0.5 \
           --enable_mkldnn=True \
           --output_dir=python_infer_output/${config}_${mode} >logs/${config}_${mode}.log 2>&1
    print_result_python
}
python_bs2(){
    mode=bs2
    python deploy/python/infer.py \
           --model_dir=./inference_model/${config} \
           --image_dir=data \
           --device=GPU \
           --run_mode=paddle \
           --threshold=0.5 \
           --batch_size=2 \
           --output_dir=python_infer_output/${config}_${mode} >logs/${config}_${mode}.log 2>&1
    print_result_python
}
python_video(){
    mode=video
    python deploy/python/infer.py \
           --model_dir=./inference_model/${config} \
           --video_file=video.mp4 \
           --device=GPU \
           --run_mode=paddle \
           --threshold=0.5 \
           --output_dir=python_infer_output/${config}_${mode} >logs/${config}_${mode}.log 2>&1
    print_result_python
}

for config in ${config_list}
do
image=demo/000000570688.jpg
if [[ -n `echo "${config_s2anet}" | grep -w "${config}"` ]];then
    image=demo/P0072__1.0__0___0.png
fi
model=`cat model_list_inference | grep ${config}`
python tools/export_model.py \
       -c configs/${model} \
       --output_dir=inference_model \
       -o weights=https://paddledet.bj.bcebos.com/models/${config}.pdparams
for mode in ${mode_list}
do
if [[ ${mode} == 'trt_int8' ]];then
    trt_calib_mode=True
else
    trt_calib_mode=False
fi
if [[ ${mode} == 'trt_int8' ]] && [[ -n `echo "${config_skip_trt8}" | grep -w "${config}"` ]];then 
    echo -e "***skip trt_int8 for ${config}"
else
    python_trt
fi
done
python_cpu
python_mkldnn
if [[ -n `echo "${config_skip_bs2}" | grep -w "${config}"` ]];then
    echo -e "***skip bs2 for ${config}"
else
    python_bs2
fi
if [[ -n `echo "${config_skip_video}" | grep -w "${config}"` ]];then
    echo -e "***skip video for ${config}"
else
    python_video
fi
done
###################################################
cd deploy/cpp
rm -rf paddle_inference
rm -rf deps/*
tar -xvf paddle_inference.tgz
mv paddle_inference_install_dir paddle_inference
sed -i "s|/path/to/paddle_inference|../paddle_inference|g" scripts/build.sh
sed -i "s|WITH_GPU=OFF|WITH_GPU=ON|g" scripts/build.sh
sed -i "s|WITH_TENSORRT=OFF|WITH_TENSORRT=ON|g" scripts/build.sh
sed -i "s|CUDA_LIB=/path/to/cuda/lib|CUDA_LIB=/usr/local/cuda/lib64|g" scripts/build.sh
if [[ "$MACHINE_TYPE" == "aarch64" ]]
then
sed -i "s|WITH_MKL=ON|WITH_MKL=OFF|g" scripts/build.sh
sed -i "s|TENSORRT_INC_DIR=/path/to/tensorrt/include|TENSORRT_INC_DIR=/usr/include/aarch64-linux-gnu|g" scripts/build.sh
sed -i "s|TENSORRT_LIB_DIR=/path/to/tensorrt/lib|TENSORRT_LIB_DIR=/usr/lib/aarch64-linux-gnu|g" scripts/build.sh
sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/aarch64-linux-gnu|g" scripts/build.sh
else
sed -i "s|TENSORRT_LIB_DIR=/path/to/tensorrt/lib|TENSORRT_LIB_DIR=/usr/local/TensorRT6-cuda10.1-cudnn7/lib|g" scripts/build.sh
sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/x86_64-linux-gnu|g" scripts/build.sh
fi
sh scripts/build.sh
cd ../..
print_result_cpp(){
    if [ $? -ne 0 ];then
        echo -e "${config}_${mode},cpp_infer,FAIL"
        cd log_err/cpp_infer
        if [ ! -d ${config} ];then
            mkdir ${config}
        fi
        cd ../..
        mv logs_cpp/${config}_${mode}.log log_err/cpp_infer/${config}
        err_sign=true
    else
        echo -e "${config}_${mode},cpp_infer,SUCCESS"
    fi
}
cpp_trt(){
    ./deploy/cpp/build/main --model_dir=inference_model/${config} --image_file=${image} --output_dir=cpp_infer_output/${config}_${mode} --device=GPU --run_mode=${mode} --threshold=0.5 --trt_calib_mode=${trt_calib_mode} >logs_cpp/${config}_${mode}.log 2>&1
print_result_cpp
}
cpp_cpu(){
mode=cpu
./deploy/cpp/build/main --model_dir=inference_model/${config} --image_file=${image} --output_dir=cpp_infer_output/${config}_${mode} --device=CPU --threshold=0.5 >logs_cpp/${config}_${mode}.log 2>&1
print_result_cpp
}
cpp_mkldnn(){
mode=mkldnn
./deploy/cpp/build/main --model_dir=inference_model/${config} --image_file=${image} --output_dir=cpp_infer_output/${config}_${mode} --device=CPU --use_mkldnn=True --threshold=0.5 >logs_cpp/${config}_${mode}.log 2>&1
print_result_cpp
}
cpp_bs2(){
mode=bs2
./deploy/cpp/build/main --model_dir=inference_model/${config} --image_dir=data --output_dir=cpp_infer_output/${config}_${mode} --device=GPU --run_mode=paddle --batch_size=2 --threshold=0.5 >logs_cpp/${config}_${mode}.log 2>&1
print_result_cpp
}
cpp_video(){
mode=video
./deploy/cpp/build/main --model_dir=inference_model/${config} --video_file=video.mp4 --output_dir=cpp_infer_output/${config}_${mode} --device=GPU --run_mode=paddle --threshold=0.5 >logs_cpp/${config}_${mode}.log 2>&1
print_result_cpp
}
for config in ${config_list_cpp}
do
image=demo/000000570688.jpg
if [[ -n `echo "${config_s2anet}" | grep -w "${config}"` ]];then
    image=demo/P0072__1.0__0___0.png
fi
for mode in ${mode_list}
do
if [[ ${mode} == 'trt_int8' ]];then
    trt_calib_mode=True
else
    trt_calib_mode=False
fi
if [[ ${mode} == 'trt_int8' ]] && [[ -n `echo "${config_skip_trt8}" | grep -w "${config}"` ]];then
    echo -e "***skip trt_int8 for ${config}"
else
    cpp_trt
fi
done
cpp_cpu
cpp_mkldnn
if [[ -n `echo "${config_skip_bs2}" | grep -w "${config}"` ]];then
    echo -e "***skip bs2 for ${config}"
else
    cpp_bs2
fi
if [[ -n `echo "${config_skip_video}" | grep -w "${config}"` ]];then
    echo -e "***skip video for ${config}"
else
    cpp_video
fi
done
if [ "${err_sign}" = true ];then
    exit 1
fi
set +x
