#!/bin/bash
# $1 本地待上传目录 $2 上传到bos路径 $3 slim commit
current_path=$(pwd)

wget_coco_data(){
    # 判断下如果没有coco
    wget https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip && unzip -o coco.zip
}

copy_coco_data(){
    # 拷贝数据到目的地
    cp -r $current_path/coco $current_path/$1/
}

upload_data() {
    # 将结果上传到bos上
    now_path=$(pwd)
    model=$1
    tar -cf ${model}_act_qat.tar ${model}_act_qat
    mv ${model}_act_qat.tar ${current_path}/${model_path}
    cd ${current_path}
    unset http_proxy && unset https_proxy
    python Bos/upload.py ${current_path}/${model_path} ${bos_path}/${slim_commit}
    python Bos/upload.py ${current_path}/${model_path} ${bos_path}
    cd ${now_path}
}
execute_yolov_modes(){
    dir=example/auto_compression/pytorch_yolo_series
    des=${dir}/dataset
    cd ${dir}
    rm -rf dataset && mkdir dataset
    models_list="yolov5s yolov6s yolov7"
    # 软链数据集
    copy_coco_data ${des}
    # 讲run中的转onnx关闭
    sed -i "s|ac.export_onnx()|#ac.export_onnx()|g" run.py
    for model in ${models_list}
    do
        # 下载数据集和预训练模型
        echo ${model}
        # 关闭配置中的
        sed -i "s|onnx_format: true|onnx_format: false|g" configs/${model}_qat_dis.yaml
        wget https://paddle-slim-models.bj.bcebos.com/act/${model}.onnx
        python run.py --config_path=./configs/${model}_qat_dis.yaml --save_dir=./${model}_act_qat/
        # 上传执行结果
        # 将原始模型重新命名
        mv ${model}.onnx  ${model}_origin.onnx
        mv ${model}_origin.onnx ./${model}_act_qat/
        upload_data ${model}
    done
}

# 其他模型
execute_ppyoloe(){
    dir=example/auto_compression/detection
    des=${dir}/dataset
    cd ${dir}
    rm -rf dataset && mkdir dataset
    # 软链数据集
    copy_coco_data ${des}
    wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar && tar -xvf ppyoloe_crn_l_300e_coco.tar
    sed -i "s|onnx_format: true|onnx_format: false|g" configs/ppyoloe_l_qat_dis.yaml
    python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir=ppyoloe_crn_l_300e_coco_act_qat
    # 上传执行结果
    mv ppyoloe_crn_l_300e_coco  ppyoloe_crn_l_300e_coco_origin
    mv ppyoloe_crn_l_300e_coco_origin ./ppyoloe_crn_l_300e_coco_act_qat/
    upload_data ppyoloe_crn_l_300e_coco
}

execute_picodet(){
    dir=example/full_quantization/picodet
    des=${dir}/dataset
    cd ${dir}
    rm -rf dataset && mkdir dataset
    # 软链数据集
    copy_coco_data ${des}
    wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar && tar -xvf picodet_s_416_coco_npu.tar
    sed -i "s|onnx_format: true|onnx_format: false|g" configs/picodet_npu_with_postprocess.yaml
    python run.py --config_path=./configs/picodet_npu_with_postprocess.yaml --save_dir='./picodet_s_416_coco_npu_act_qat/'
    # 上传执行结果
    mv picodet_s_416_coco_npu  picodet_s_416_coco_npu_origin
    mv picodet_s_416_coco_npu_origin ./picodet_s_416_coco_npu_act_qat/
    upload_data picodet_s_416_coco_npu
}


main(){
    # 总调度入口
    wget_coco_data
    execute_yolov_modes
    cd $current_path
    execute_ppyoloe
    cd $current_path
    execute_picodet
    cd $current_path
}

model_path=$1
bos_path=$2
slim_commit=$3
main
