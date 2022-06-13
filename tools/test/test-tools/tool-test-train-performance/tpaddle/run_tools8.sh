#!/usr/bin/env bash
#工具8（训练耗时测试工具）用例数量：5
unset GREP_OPTIONS
export repo_path=$PWD
export tools_path=$PWD
export TRAIN_LOG_DIR=${tools_path}/tool_log
rm -rf ${TRAIN_LOG_DIR}
mkdir -p ${TRAIN_LOG_DIR}
cards=${2:-"1"}
if [ $cards = "1" ];then
     export CUDA_VISIBLE_DEVICES=0
elif [ $cards = "8" ];then
     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
     echo "cards error"
     exit 1;
fi
# 用例1 dynamic_ResNet50_vd 单卡
dynamic_ResNet50_vd_bs64_1_1_sp(){
num_gpu_devices=1;    # 运行卡数
log_file=${TRAIN_LOG_DIR}/dynamic_ResNet50_vd_bs64_1_sp  # 模型log
model_name=ResNet50_vd_bs64 # 模型名
mission_name="图像分类" # 模型类别
direction_id=0        # 任务所属方向，0：CV，1：NLP，2：Rec  (必填)
skip_steps=2         # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
run_mode=sp           # 该模型运行的模式,sp:单进程,mp:多进程                                          (必填)
keyword="ips:"        # 解析日志，筛选出数据所在行的关键字                                             (必填)
base_batch_size=64    # 模型运行单卡 batch_size                                                    (必填)
index=1               # 1表示计算log中速度  (必填)
ips_unit="images/s"   # 解析日志，按照分隔符分割后形成的数组索引                                         (必填)
FrameName=paddlepaddle # 框架名
# 运行模型
#export CUDA_VISIBLE_DEVICES=0;
bash run_ResNet50_vd.sh ${num_gpu_devices}
python analysis.py \
            --filename "${log_file}" \
            --model_name "${model_name}" \
            --mission_name "${mission_name}" \
            --direction_id ${direction_id} \
            --run_mode ${run_mode} \
            --keyword "${keyword}" \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --gpu_num ${num_gpu_devices} \
            --index ${index} \
            --ips_unit ${ips_unit} \
            --FrameName ${FrameName}

}
# 用例2 dynamic_ResNet50_vd 8卡多进程
dynamic_ResNet50_vd_bs64_1_8_mp(){
log_file=${TRAIN_LOG_DIR}/dynamic_ResNet50_vd_bs64_8_mp
num_gpu_devices=8;
model_name=ResNet50_vd_bs64
mission_name="图像分类"
direction_id=0
skip_steps=2
run_mode=mp
keyword="ips:"
base_batch_size=64
index=1
ips_unit="images/s"
FrameName=paddlepaddle
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
bash run_ResNet50_vd.sh ${num_gpu_devices}
python analysis.py \
            --filename "${log_file}" \
            --model_name "${model_name}" \
            --mission_name "${mission_name}" \
            --direction_id ${direction_id} \
            --run_mode ${run_mode} \
            --keyword "${keyword}" \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --gpu_num ${num_gpu_devices} \
            --index ${index} \
            --ips_unit ${ips_unit} \
            --FrameName ${FrameName}
}

# 用例3 dynamic_MobileNetV1_bs128_1_sp
dynamic_MobileNetV1_bs128_1_sp(){
log_file=${TRAIN_LOG_DIR}/dynamic_MobileNetV1_bs128_1_sp
num_gpu_devices=1;
model_name=MobileNetV1_bs128
mission_name="图像分类"
direction_id=0
skip_steps=2
run_mode=sp
keyword="ips:"
base_batch_size=128
index=1
ips_unit="images/s"
FrameName=paddlepaddle
#export CUDA_VISIBLE_DEVICES=0;
bash run_MobileNetV1.sh ${num_gpu_devices}
python analysis.py \
            --filename "${log_file}" \
            --model_name "${model_name}" \
            --mission_name "${mission_name}" \
            --direction_id ${direction_id} \
            --run_mode ${run_mode} \
            --keyword "${keyword}" \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --gpu_num ${num_gpu_devices} \
            --index ${index} \
            --ips_unit ${ips_unit} \
            --FrameName ${FrameName}
}

# 用例4 dynamic_MobileNetV1_bs128_8_mp 8卡多进程
dynamic_MobileNetV1_bs128_8_mp(){
log_file=${TRAIN_LOG_DIR}/dynamic_MobileNetV1_bs128_8_mp
num_gpu_devices=8;
model_name=MobileNetV1_bs128
mission_name="图像分类"
direction_id=0
skip_steps=2
run_mode=mp
keyword="ips:"
base_batch_size=128
index=1
ips_unit="images/s"
FrameName=paddlepaddle
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
bash run_MobileNetV1.sh ${num_gpu_devices}
python analysis.py \
            --filename "${log_file}" \
            --model_name "${model_name}" \
            --mission_name "${mission_name}" \
            --direction_id ${direction_id} \
            --run_mode ${run_mode} \
            --keyword "${keyword}" \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --gpu_num ${num_gpu_devices} \
            --index ${index} \
            --ips_unit ${ips_unit} \
            --FrameName ${FrameName}
}

# 用例5 dynamic_yolov3_bs8_1_sp
dynamic_yolov3_bs8_1_sp(){
log_file=${TRAIN_LOG_DIR}/dynamic_yolov3_bs8_1_sp
num_gpu_devices=1;
model_name=yolov3_bs8
mission_name="目标检测"
direction_id=0
skip_steps=1
run_mode=sp
keyword="ips:"
base_batch_size=8
index=1
ips_unit="images/s"
FrameName=paddlepaddle
#export CUDA_VISIBLE_DEVICES=0;
bash run_yolov3.sh ${num_gpu_devices}
python analysis.py \
            --filename "${log_file}" \
            --model_name "${model_name}" \
            --mission_name "${mission_name}" \
            --direction_id ${direction_id} \
            --run_mode ${run_mode} \
            --keyword "${keyword}" \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --gpu_num ${num_gpu_devices} \
            --index ${index} \
            --ips_unit ${ips_unit} \
            --FrameName ${FrameName}
}
# 用例6 dynamic_yolov3_bs8_8_mp 8卡多进程
dynamic_yolov3_bs8_8_mp(){
log_file=${TRAIN_LOG_DIR}/dynamic_yolov3_bs8_8_mp
num_gpu_devices=8;
model_name=yolov3_bs8
mission_name="目标检测"
direction_id=0
skip_steps=1
run_mode=mp
keyword="ips:"
base_batch_size=8
index=1
ips_unit="images/s"
FrameName=paddlepaddle
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
bash run_yolov3.sh ${num_gpu_devices}
python analysis.py \
            --filename "${log_file}" \
            --model_name "${model_name}" \
            --mission_name "${mission_name}" \
            --direction_id ${direction_id} \
            --run_mode ${run_mode} \
            --keyword "${keyword}" \
            --base_batch_size ${base_batch_size} \
            --skip_steps ${skip_steps} \
            --gpu_num ${num_gpu_devices} \
            --index ${index} \
            --ips_unit ${ips_unit} \
            --FrameName ${FrameName}
}



if [ $1 = "ResNet50_vd" ];then
    cd ${repo_path}/PaddleClas/
    rm -rf run_ResNet50_vd.sh;
    cp ${tools_path}/run_ResNet50_vd.sh ./
    cp ${tools_path}/analysis.py ./
    if [ $2 = "1" ];then
        dynamic_ResNet50_vd_bs64_1_1_sp
    elif [ $2 = "8" ];then
        dynamic_ResNet50_vd_bs64_1_8_mp
    else
        echo "cards error"
    fi
elif [ $1 = "MobileNetV1" ];then
    cd ${repo_path}/PaddleClas/
    rm -rf run_MobileNetV1.sh
    cp ${tools_path}/run_MobileNetV1.sh ./
    cp ${tools_path}/analysis.py ./

    if [ $2 = "1" ];then
        dynamic_MobileNetV1_bs128_1_sp
    elif [ $2 = "8" ];then
        dynamic_MobileNetV1_bs128_8_mp
    else
        echo "cards error"
    fi
elif [ $1 = "yolov3" ];then
    cd ${repo_path}/PaddleDetection/
    rm -rf run_yolov3.sh
    cp ${tools_path}/run_yolov3.sh ./
    cp ${tools_path}/analysis.py ./
    if [ $2 = "1" ];then
        dynamic_yolov3_bs8_1_sp
    elif [ $2 = "8" ];then
        dynamic_yolov3_bs8_8_mp
    else
        echo "cards error"
    fi
else
    echo "model_name error"
    exit 1;
fi
