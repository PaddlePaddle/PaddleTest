#!/usr/bin/env bash
####################################
# set paddlenlp
set -x
nlp1_build (){
    echo -e "\033[35m ---- only install paddlenlp \033[0m"
    python -m pip install -U paddlenlp
}
nlp2_build (){
    echo -e "\033[35m ---- build and install paddlenlp  \033[0m"
    rm -rf build/
    rm -rf paddlenlp.egg-info/
    rm -rf dist/

    python -m pip install -r requirements.txt
    python setup.py bdist_wheel
    python -m pip install dist/paddlenlp****.whl
}
$2;
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
pip list
set +x
####################################
# for logs env
export nlp_dir=/workspace
if [ -d "/workspace/logs" ];then
    rm -rf /workspace/logs;
fi
mkdir /workspace/logs
export log_path=/workspace/logs
####################################
# run api case
python test_trainsformers.py
P0case_EXCODE=$? || true
####################################
# upload log to bos
cd ${nlp_dir}/
cp -r /ssd1/paddlenlp/bos/* ./
tar -zcvf logs_models.tar logs/
mkdir upload && mv logs_models.tar upload
python upload.py upload
####################################
echo -e "\033[35m ---- result: \033[0m"
echo -e "\033[35m ---- P0case_EXCODE: $P0case_EXCODE \033[0m"
if [ $P0case_EXCODE -ne 0 ] ; then
    cd logs
    FF=`ls *_FAIL*|wc -l`
    echo -e "\033[31m ---- P0case failed number: ${FF} \033[0m"
    ls *_FAIL*
    exit $P0case_EXCODE
else
    echo -e "\033[32m ---- P0case Success \033[0m"
fi
