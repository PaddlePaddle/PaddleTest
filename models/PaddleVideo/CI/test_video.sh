#!/bin/bash
#prepare env
mkdir run_env_py37
ln -s $(which python3.7) run_env_py37/python
ln -s $(which pip3.7) run_env_py37/pip
export PATH=$(pwd)/run_env_py37:${PATH}
export http_proxy=${proxy}
export https_proxy=${proxy}
export no_proxy=bcebos.com
python -m pip install pip==20.2.4 --ignore-installed
pip install -r requirements.txt --ignore-installed
python -m pip  install ${paddle_whl} --no-cache-dir
echo -e '*****************paddle_version*****'
python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************video_version****'
git rev-parse HEAD
#create log dir
if [ -d "log" ];then rm -rf log
fi
mkdir log
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err
#prepare data
if [ -d "data/ucf101" ]; then rm -rf data/ucf101
fi
ln -s ${file_path}/data/ucf101 data/ucf101
if [ -d "data/k400" ]; then rm -rf data/k400
fi
ln -s ${file_path}/data/k400 data/k400
cd data
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
cd ..
print_result(){
    if [ $? -ne 0 ];then
        echo -e "${model},${mode},FAIL"
        cd log_err
        if [ ! -d ${model} ];then
            mkdir ${model}
        fi
        cd ../${model_path}
        mv log/${model}/${model}_${mode}.log log_err/${model}/
        err_sign=true
        #exit 1
    else
        echo -e "${model},${mode},SUCCESS"
    fi
}
TRAIN(){
    export CUDA_VISIBLE_DEVICES=$cudaid2
    mode=train
    python -m paddle.distributed.launch main.py \
              -c ${config} \
              -o epochs=1 >log/${model}/${model}_train.log 2>&1 
    print_result    
}
EVAL(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=eval
    python main.py \
           --test \
           -c ${config} \
           -w output/${model}/${model}_epoch_00001.pdparams >log/${model}/${model}_eval.log 2>&1
    print_result
}
EXPORT(){
    mode=export
    python tools/export_model.py \
           -c ${config} \
           -p output/${model}/${model}_epoch_00001.pdparams \
           -o inference/${model} >log/${model}/${model}_export.log 2>&1
    print_result
}
INFER(){
    mode=infer
    python tools/predict.py \
           --input_file data/ucf101/videos/PizzaTossing/v_PizzaTossing_g10_c02.avi \
           --config ${config} \
           --model_file inference/${model}/${model}.pdmodel \
           --params_file inference/${model}/${model}.pdiparams \
           --use_gpu=True \
           --use_tensorrt=False >log/${model}/${model}_infer.log 2>&1
    print_result
}
TRT(){
    mode=trt
    python tools/predict.py \
           --input_file data/ucf101/videos/PizzaTossing/v_PizzaTossing_g10_c02.avi \
           --config ${config} \
           --model_file inference/${model}/${model}.pdmodel \
           --params_file inference/${model}/${model}.pdiparams \
           --use_gpu=True \
           --use_tensorrt=True \
           --batch_size=${trt_bs} >log/${model}/${model}_trt.log 2>&1
    print_result
}
model_list='TSM ppTSN'
for model in ${model_list}
do
typeset -l model_small
model_small=${model}
if [[ ${model} == 'TSM' ]];then
trt_bs=8
else
trt_bs=2
fi
config=`cat model_list_video | grep ${model_small}`
cd log
mkdir ${model}
cd ..
TRAIN
EVAL
EXPORT
INFER
TRT
done
if [ "${err_sign}" = true ];then
    exit 1
fi
