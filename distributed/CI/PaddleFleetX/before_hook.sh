#!/usr/bin/env bash
set -e

export fleetx_path=/paddle/PaddleFleetX
export data_path=/fleetx_data
export log_path=/paddle/log_fleetx
mkdir -p ${log_path}

unset CUDA_VISIBLE_DEVICES

function kill_fleetx_process(){
  kill -9 `ps -ef|grep run_pretrain.py|awk '{print $2}'`
  kill -9 `ps -ef|grep run_generation.py|awk '{print $2}'`
}

function requirements() {
    echo "=============================paddle commit============================="
    python -c "import paddle;print(paddle.__git_commit__)"

    # if [[ ${AGILE_COMPILE_BRANCH} =~ "develop" ]];then
    #     cd /paddle
    #     echo " ---------- PaddleFleetX develop Slim---------- "
    #     export https_proxy=${proxy}
    #     export http_proxy=${proxy}
    #     sed -i "s/git+https/#git+https/g" ./PaddleFleetX/requirements.txt
    #     git clone --depth=1 -b develop https://github.com/PaddlePaddle/PaddleSlim.git
    #     rm -rf /usr/local/lib/python3.7/dist-packages/paddleslim*
    #     python -m pip uninstall paddleslim -y
    #     cd PaddleSlim
    #     python -m pip install -r requirements.txt --force-reinstall
    #     python setup.py install
    #     cd -
    #     unset http_proxy && unset https_proxy
    # fi

    # install requirements
    cd ${fleetx_path}
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    python -m pip install -r requirements.txt --force-reinstall
}

function download() {
    cd ${fleetx_path}

    rm -rf ckpt
    if [[ -e ${data_path}/ckpt/PaddleFleetX_GPT_345M_220826 ]]; then
        echo "ckpt/PaddleFleetX_GPT_345M_220826 downloaded"
    else
        # download ckpt
        mkdir -p ${data_path}/ckpt
        wget -O ${data_path}/ckpt/GPT_345M.tar.gz \
            https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
        tar -xzf ${data_path}/ckpt/GPT_345M.tar.gz -C ${data_path}/ckpt
        rm -rf ${data_path}/ckpt/GPT_345M.tar.gz
    fi

    if [[ -e ${data_path}/ckpt/model.pdparams ]]; then
        echo "ckpt/imagenet2012-ViT-B_16-224.pdparams downloaded"
    else
        # download ckpt
        mkdir -p ${data_path}/ckpt
        wget -O ${data_path}/ckpt/model.pdparams \
            https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams
    fi
    ln -s ${data_path}/ckpt ${fleetx_path}/ckpt

    rm -rf data
    if [[ -e ${data_path}/data ]]; then
        echo "data downloaded"
    else
        # download data
        mkdir ${data_path}/data;
        wget -O ${data_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
        wget -O ${data_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
    fi
    cp -r ${data_path}/data ${fleetx_path}/

    rm -rf dataset
    if [[ -e ${data_path}/dataset/wikitext_103_en ]]; then
        echo "dataset/wikitext_103_en downloaded"
    else
        # download dataset/wikitext_103_en
        mkdir ${data_path}/dataset/wikitext_103_en;
        wget -O ${data_path}/dataset/wikitext_103_en/wikitext-103-en.txt http://fleet.bj.bcebos.com/datasets/gpt/wikitext-103-en.txt
    fi
    if [[ -e ${data_path}/dataset/ernie ]]; then
        echo "dataset/ernie downloaded"
    else
        # download dataset/ernie
        mkdir -p ${data_path}/dataset/ernie;
        wget -O ${data_path}/dataset/ernie/cluecorpussmall_14g_1207_ids_part0 https://paddlefleetx.bj.bcebos.com/model/nlp/ernie/cluecorpussmall_14g_1207_ids_part0
        wget -O ${data_path}/dataset/ernie/cluecorpussmall_14g_1207_ids_part1 https://paddlefleetx.bj.bcebos.com/model/nlp/ernie/cluecorpussmall_14g_1207_ids_part1
        cat ${data_path}/dataset/ernie/cluecorpussmall_14g_1207_ids_part* &> ${data_path}/dataset/ernie/cluecorpussmall_14g_1207_ids.npy
        wget -O ${data_path}/dataset/ernie/cluecorpussmall_14g_1207_idx.npz https://paddlefleetx.bj.bcebos.com/model/nlp/ernie/cluecorpussmall_14g_1207_idx.npz
    fi
    cp -r ${data_path}/dataset ${fleetx_path}/

    rm -rf cc12m_base64
    if [[ -e ${data_path}/cc12m_base64 ]]; then
        echo "cc12m_base64 downloaded"
    else
        # download cc12m_base64
        wget -O ${data_path}/cc12m_base64.tar https://fleetx.bj.bcebos.com/datasets/cc12m_base64.tar
        tar xf ${data_path}/cc12m_base64.tar -C ${data_path}/
        rm -rf ${data_path}/cc12m_base64.tar
    fi
    ln -s ${data_path}/cc12m_base64 ${fleetx_path}/cc12m_base64
}


main() {
    requirements
    download
}

main$@
