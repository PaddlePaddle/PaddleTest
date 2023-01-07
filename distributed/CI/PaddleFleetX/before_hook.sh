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

    if [[ ${AGILE_COMPILE_BRANCH} =~ "develop" ]];then
        cd /paddle
        echo " ---------- PaddleFleetX develop Slim---------- "
        export https_proxy=${proxy}
        export http_proxy=${proxy}
        sed -i "s/git+https/#git+https/g" ./PaddleFleetX/requirements.txt
        python -m pip uninstall paddleslim -y
        python -m pip install https://paddle-qa.bj.bcebos.com/PaddleSlim/paddleslim-0.0.0.dev0-py3-none-any.whl --no-cache-dir --force-reinstall --no-dependencies
        unset http_proxy && unset https_proxy

        # echo " ---------- PaddleFleetX develop paddlenlp---------- "
        # export https_proxy=${proxy}
        # export http_proxy=${proxy}
        # sed -i "s/paddlenlp/#paddlenlp/g" ./PaddleFleetX/requirements.txt
        # python -m pip install paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html --force-reinstall
        # unset http_proxy && unset https_proxy
    fi

    # install requirements
    cd ${fleetx_path}
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    python -m pip install -r requirements.txt --force-reinstall

    python -m pip list|grep paddle
}

function download() {
    cd ${fleetx_path}

    rm -rf ckpt
    if [[ -e ${data_path}/ckpt/PaddleFleetX_GPT_345M_220826 ]]; then
        echo "ckpt/PaddleFleetX_GPT_345M_220826 downloaded"
    else
        # download ckpt for gpt
        mkdir -p ${data_path}/ckpt
        wget -O ${data_path}/ckpt/GPT_345M.tar.gz \
            https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
        tar -xzf ${data_path}/ckpt/GPT_345M.tar.gz -C ${data_path}/ckpt
        rm -rf ${data_path}/ckpt/GPT_345M.tar.gz
    fi

    if [[ -e ${data_path}/ckpt/model.pdparams ]]; then
        echo "ckpt/imagenet2012-ViT-B_16-224.pdparams downloaded"
    else
        # download ckpt for vit
        mkdir -p ${data_path}/ckpt
        wget -O ${data_path}/ckpt/model.pdparams \
            https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams
    fi
    ln -s ${data_path}/ckpt ${fleetx_path}/ckpt

    rm -rf data
    if [[ -e ${data_path}/data ]]; then
        echo "data downloaded"
    else
        # download data for gpt
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

    rm -rf ./projects/imagen/t5
    if [[ -e ${data_path}/t5 ]]; then
        echo "imagen/t5 downloaded"
    else
        # download t5 model
        mkdir -p ${data_path}/t5/t5-11b/ && cd ${data_path}/t5/t5-11b/
        wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.0
        wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.1
        wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.2
        wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.3
        wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.4
        wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/config.json
        wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/spiece.model
        wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/tokenizer.json
        cat t5.pd.tar.gz.* |tar -xf - 
        rm -rf t5.pd.tar.gz.*
        cd -
    fi
    ln -s ${data_path}/t5 ${fleetx_path}/projects/imagen/t5

    rm -rf ./projects/imagen/cache
    if [[ -e ${data_path}/cache ]]; then
        echo "imagen/cache downloaded"
    else
        # download debertav2 for imagen
        mkdir -p ${data_path}/cache && cd ${data_path}/cache
        wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/config.json
        wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/spm.model
        wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/tokenizer_config.json
        wget https://fleetx.bj.bcebos.com/DebertaV2/debertav2.pd.tar.gz.0
        wget https://fleetx.bj.bcebos.com/DebertaV2/debertav2.pd.tar.gz.1
        tar debertav2.pd.tar.gz.* | tar -xf -  
        rm -rf debertav2.pd.tar.gz.*
        cd -
    fi
    ln -s ${data_path}/cache ${fleetx_path}/projects/imagen/cache

    rm -rf part-00079
    if [[ -e ${data_path}/part-00079 ]]; then
        echo "part-00079 downloaded"
    else
        # download part-00079 for imagen
        wget -O ${data_path}/part-00079 https://paddlefleetx.bj.bcebos.com/data/laion400m/part-00079
    fi
    cp ${data_path}/part-00079 ${fleetx_path}/projects/imagen

    rm -rf wikitext-103
    if [[ -e ${data_path}/wikitext-103 ]]; then
        echo "wikitext-103 downloaded"
    else
        # download wikitext-103 for gpt eval
        wget -O ${data_path}/wikitext-103-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
        unzip -q ${data_path}/wikitext-103-v1.zip
        rm -rf ${data_path}/wikitext-103-v1.zip
    fi
    ln -s ${data_path}/wikitext-103 ${fleetx_path}/wikitext-103

    rm -rf lambada_test.jsonl
    if [[ -e ${data_path}/lambada_test.jsonl ]]; then
        echo "lambada_test.jsonl downloaded"
    else
        # download lambada_test.jsonl for gpt eval
        wget -O ${data_path}/lambada_test.jsonl https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl
    fi
    cp ${data_path}/lambada_test.jsonl ${fleetx_path}/
}


main() {
    requirements
    download
}

main$@
