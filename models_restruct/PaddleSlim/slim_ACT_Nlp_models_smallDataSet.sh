#!/bin/bash
current_path=$(pwd)
echo "当前路径是：$current_path"

upload_data() {
    # 将结果上传到bos上
    echo "update quant models!"
}
execute_ERNIE3_Medium(){
    dir=example/auto_compression/nlp
    cd ${dir}
    wget https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/AFQMC.tar && tar -xf AFQMC.tar
    python run.py --config_path='./configs/ernie3.0/afqmc.yaml' --save_dir='./save_ernie3_afqmc_new_cablib/'
}

execute_pp_minilm(){
    dir=example/auto_compression/nlp
    cd ${dir}
    wget https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar && tar -xf afqmc.tar
    python run.py --config_path='./configs/pp-minilm/auto/afqmc.yaml' --save_dir='./save_ppminilm_afqmc_new_calib'
    
}

execute_Bert_base_cased(){
    dir=example/auto_compression/pytorch_huggingface
    cd ${dir}
    wget https://paddle-slim-models.bj.bcebos.com/act/x2paddle_cola.tar && tar -xf x2paddle_cola.tar
    python run.py --config_path=./configs/cola.yaml --save_dir='./x2paddle_cola_new_calib/'
}

main(){
    # 总调度入口
    execute_ERNIE3_Medium
    cd $current_path
    execute_pp_minilm
    cd $current_path
    execute_Bert_base_cased
    cd $current_path
}

main
