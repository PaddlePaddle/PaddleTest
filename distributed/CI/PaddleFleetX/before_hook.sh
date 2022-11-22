#!/usr/bin/env bash
set -e

export fleetx_path=/paddle/PaddleFleetX
export cp_path=/home/FleetX_CI
export log_path=/paddle/log
mkdir -p ${log_path}

unset CUDA_VISIBLE_DEVICES

function kill_fleetx_process(){
  kill -9 `ps -ef|grep run_pretrain.py|awk '{print $2}'`
  kill -9 `ps -ef|grep run_generation.py|awk '{print $2}'`
}

function before_hook() {
    echo "=============================paddle commit============================="
    python -c "import paddle;print(paddle.__git_commit__)"

    # install requirements
    cd ${fleetx_path}
    export http_proxy=http://172.19.57.45:3128
    export https_proxy=http://172.19.57.45:3128
    export no_proxy=bcebos.com
    python -m pip install -r requirements.txt --force-reinstall

    mkdir -p ckpt
    if [[ -e ${cp_path}/ckpt/PaddleFleetX_GPT_345M_220826 ]]; then
        echo "ckpt/PaddleFleetX_GPT_345M_220826 downloaded"
    else
        # download ckpt
        mkdir -p ${cp_path}/ckpt
        wget -O ${cp_path}/ckpt/GPT_345M.tar.gz \
            https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
        tar -xzf ${cp_path}/ckpt/GPT_345M.tar.gz -C ${cp_path}/ckpt
        rm -rf ${cp_path}/ckpt/GPT_345M.tar.gz
    fi
    cp -r ${cp_path}/ckpt/PaddleFleetX_GPT_345M_220826 ${fleetx_path}/ckpt/

    if [[ -e ${cp_path}/ckpt/model.pdparams ]]; then
        echo "ckpt/imagenet2012-ViT-B_16-224.pdparams downloaded"
    else
        # download ckpt
        mkdir -p ${cp_path}/ckpt
        wget -O ${cp_path}/ckpt/model.pdparams \
            https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams
    fi
    cp -r ${cp_path}/ckpt/model.pdparams ${fleetx_path}/ckpt/

    if [[ -e ${cp_path}/data ]]; then
        echo "data downloaded"
    else
        # download data
        mkdir ${cp_path}/data;
        wget -O ${cp_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
        wget -O ${cp_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
    fi
    cp -r ${cp_path}/data ${fleetx_path}/

    if [[ -e ${cp_path}/dataset/wikitext_103_en ]]; then
        echo "dataset/wikitext_103_en downloaded"
    else
        # download dataset/wikitext_103_en
        mkdir ${cp_path}/dataset/wikitext_103_en;
        wget -O ${cp_path}/dataset/wikitext_103_en/wikitext-103-en.txt http://fleet.bj.bcebos.com/datasets/gpt/wikitext-103-en.txt
    fi
    if [[ -e ${cp_path}/dataset/ernie ]]; then
        echo "dataset/ernie downloaded"
    else
        # download dataset/ernie
        unset https_proxy && unset http_proxy
        mkdir -p ${cp_path}/dataset/ernie;
        wget -O dataset/ernie/cluecorpussmall_14g_1207_ids.npy http://10.255.129.12:8811/cluecorpussmall_14g_1207_ids.npy
        wget -O dataset/ernie/cluecorpussmall_14g_1207_idx.npz http://10.255.129.12:8811/cluecorpussmall_14g_1207_idx.npz
    fi
    cp -r ${cp_path}/dataset ${fleetx_path}/

    cp /paddle/PaddleTest/distributed/CI/PaddleFleetX/generation_base.txt ${fleetx_path}/
}

main() {
    before_hook
}

main$@
