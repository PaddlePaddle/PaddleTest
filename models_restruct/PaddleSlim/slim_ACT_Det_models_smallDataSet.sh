#!/bin/bash
current_path=$(pwd)
echo "当前路径是：$current_path"

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
    echo "update quant models!"
}
execute_yolov_modes(){
    dir=example/auto_compression/pytorch_yolo_series
    des=${dir}/dataset
    cd ${dir}
    rm -rf dataset && mkdir dataset
    models_list="yolov5s yolov6s yolov7"
    # 软链数据集
    copy_coco_data ${des}
    for model in ${models_list}
    do
        # 下载数据集和预训练模型
        echo ${model}
        wget https://paddle-slim-models.bj.bcebos.com/act/${model}.onnx
        python run.py --config_path=./configs/${model}_qat_dis.yaml --save_dir=./${model}_quant/
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
    python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir=ppyoloe_crn_l_300e_coco_quant
}

execute_picodet(){
    dir=example/full_quantization/picodet
    des=${dir}/dataset
    cd ${dir}
    rm -rf dataset && mkdir dataset
    # 软链数据集
    copy_coco_data ${des}
    wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar && tar -xvf picodet_s_416_coco_npu.tar
    python run.py --config_path=./configs/picodet_npu_with_postprocess.yaml --save_dir='./picodet_s_416_coco_npu_quant/' 
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

main
