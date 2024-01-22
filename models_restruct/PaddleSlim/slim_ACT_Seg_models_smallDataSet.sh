#!/bin/bash
current_path=$(pwd)
echo "当前路径是：$current_path"

wget_data(){
    # 判断下如果没有coco
    wget https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar && tar -xf mini_cityscapes.tar
    mv mini_cityscapes cityscapes
}

copy_data(){
    # 原地址link到目的地址
    cp -r $current_path/cityscapes $current_path/$1/
}


execute_Deeplabv3_ResNet50(){
    dir=example/auto_compression/semantic_segmentation
    des=${dir}/data
    cd ${dir}
    # 软链数据集
    copy_data ${des}
    wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-Deeplabv3-ResNet50.zip && unzip -o RES-paddle2-Deeplabv3-ResNet50.zip
    python run.py --config_path='./configs/deeplabv3/deeplabv3_qat.yaml' --save_dir='./deeplabv3_qat'
}

execute_PP_Liteseg(){
    dir=example/auto_compression/detection
    des=${dir}/data
    cd ${dir}
    # 软链数据集
    copy_data ${des}
    wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-PPLIteSegSTDC1.zip && unzip RES-paddle2-PPLIteSegSTDC1.zip
    python run.py --config_path='./configs/pp_liteseg/pp_liteseg_qat.yaml' --save_dir='./pp_liteseg_qat'
}

execute_Unet(){
    dir=example/auto_compression/detection
    des=${dir}/data
    cd ${dir}
    wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-UNet.zip && unzip -q RES-paddle2-UNet.zip
    python run.py --config_path='./configs/unet/unet_qat.yaml' --save_dir='./unet_qat'
}

execute_Hrnet(){
    dir=example/auto_compression/detection
    des=${dir}/data
    cd ${dir}
    wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-HRNetW18-Seg.zip && unzip -o RES-paddle2-HRNetW18-Seg.zip
    python run.py --config_path='./configs/hrnet/hrnet_qat.yaml' --save_dir='./hrnet_qat'
}

execute_HumanSeg(){
    dir=example/auto_compression/detection
    des=${dir}/data
    cd ${dir}
    wget https://paddle-qa.bj.bcebos.com/PaddleSlim/portrait14k.tar.gz
    tar -xf portrait14k.tar.gz
    mv portrait14k humanseg
    wget https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz && tar -xf ppseg_lite_portrait_398x224_with_softmax.tar.gz
    python run.py --config_path='./configs/pp_humanseg/pp_humanseg_qat.yaml' --save_dir='./pp_humanseg_qat'
}

main(){
    # 总调度入口
    wget_data
    execute_Deeplabv3_ResNet50
    cd ${current_path}
    execute_PP_Liteseg
    cd ${current_path}
    execute_Unet
    cd ${current_path}
    execute_Hrnet
    cd ${current_path}
    execute_HumanSeg
    cd ${current_path}
}

main
