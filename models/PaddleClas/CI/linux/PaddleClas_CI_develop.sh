unset GREP_OPTIONS
echo ${cudaid1}
echo ${cudaid2}
echo ${model_flag}
echo ${Data_path}
echo ${Project_path}
echo ${paddle_compile}
echo ${py_paddle_flag}
export CUDA_VISIBLE_DEVICES=${cudaid2}
export FLAGS_use_virtual_memory_auto_growth=1 #wanghuan 优化显存
export FLAGS_use_stream_safe_cuda_allocator=1 #zhengtianyu 环境变量测试功能性
export PADDLE_LOG_LEVEL=debug  #输出多卡log信息
export FLAGS_enable_gpu_memory_usage_log=1 #输出显卡占用峰值

echo "path before"
pwd
if [[ ${model_flag} =~ 'CE' ]]; then
    cd ${Project_path}
    echo "path after"
    pwd
    export FLAGS_cudnn_deterministic=True
    # export FLAGS_enable_eager_mode=1 #验证天宇 220329 pr  在任务重插入
    unset FLAGS_enable_eager_mode
    unset FLAGS_use_virtual_memory_auto_growth
    unset FLAGS_use_stream_safe_cuda_allocator
fi

#check 新动态图
echo "set FLAGS_enable_eager_mode"
echo $FLAGS_enable_eager_mode

# <-> model_flag CI是效率云 step0是clas分类 step1是clas分类 step2是clas分类 step3是识别，CI_all是全部都跑
#     pr是TC，clas是分类，rec是识别，single是单独模型debug
# <-> pr_num   随机跑pr的模型数
# <-> python   python版本
# <-> cudaid   cuda卡号
# <-> paddle_compile   paddle包地址
# <-> Data_path   数据路径
#$1 <-> 自己定义的  single_yaml_debug  单独模型yaml字符
#model_clip 根据日期单双数决定剪裁
echo "######  ----ln  data-----"
rm -rf dataset
ln -s ${Data_path} dataset
ls dataset |head -n 2
cd deploy
ln -s ${Data_path}/rec_demo/* .
cd ..

if ([[ ${model_flag} =~ 'pr' ]] || [[ ${model_flag} =~ 'single' ]]) &&  [[ ! ${py_paddle_flag} ]]; then #model_flag
    echo "######  model_flag pr"

    echo "######  ---py37  env -----"
    # rm -rf /usr/local/python2.7.15/bin/python
    # rm -rf /usr/local/bin/python
    # export PATH=/usr/local/bin/python:${PATH}
    case ${python} in #python
    36)
    # ln -s /usr/local/bin/python3.6 /usr/local/bin/python
    mkdir run_env_py36;
    ln -s $(which python3.6) run_env_py36/python;
    ln -s $(which pip3.6) run_env_py36/pip;
    export PATH=$(pwd)/run_env_py36:${PATH};
    ;;
    37)
    # ln -s /usr/local/bin/python3.7 /usr/local/bin/python
    mkdir run_env_py37;
    ln -s $(which python3.7) run_env_py37/python;
    ln -s $(which pip3.7) run_env_py37/pip;
    export PATH=$(pwd)/run_env_py37:${PATH};
    ;;
    38)
    # ln -s /usr/local/bin/python3.8 /usr/local/bin/python
    mkdir run_env_py38;
    ln -s $(which python3.8) run_env_py38/python;
    ln -s $(which pip3.8) run_env_py38/pip;
    export PATH=$(pwd)/run_env_py38:${PATH};
    ;;
    39)
    # ln -s /usr/local/bin/python3.9 /usr/local/bin/python
    mkdir run_env_py39;
    ln -s $(which python3.9) run_env_py39/python;
    ln -s $(which pip3.9) run_env_py39/pip;
    export PATH=$(pwd)/run_env_py39:${PATH};
    ;;
    310)
    # ln -s /usr/local/bin/python3.9 /usr/local/bin/python
    mkdir run_env_py310;
    ln -s $(which python3.10) run_env_py310/python;
    ln -s $(which pip3.10) run_env_py310/pip;
    export PATH=$(pwd)/run_env_py310:${PATH};
    ;;
    esac

    unset http_proxy
    unset https_proxy
    echo "######  ----install  paddle-----"
    python -m pip install --ignore-installed --upgrade \
        pip -i https://mirror.baidu.com/pypi/simple
    python -m pip uninstall paddlepaddle-gpu -y
    python -m pip install ${paddle_compile} -i https://mirror.baidu.com/pypi/simple #paddle_compile

fi

# paddle
echo "######  paddle version"
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

echo "######  clas version"
git rev-parse HEAD

# python
python -c 'import sys; print(sys.version_info[:])'
echo "######  python version"

#system
if [ -d "/etc/redhat-release" ]; then
   echo "######  system centos"
else
   echo "######  system linux"
fi

unset http_proxy
unset https_proxy
# env
export FLAGS_fraction_of_gpu_memory_to_use=0.8
# python -m pip install --ignore-installed --upgrade \
#    setuptools==59.5.0 -i https://mirror.baidu.com/pypi/simple #before install bcolz
python -m pip install --ignore-installed --upgrade \
   pip -i https://mirror.baidu.com/pypi/simple
python -m pip install  --ignore-installed paddleslim \
   -i https://mirror.baidu.com/pypi/simple
# python -m pip install --ignore-installed dataset/visualdl-2.2.1-py3-none-any.whl \
#    -i https://mirror.baidu.com/pypi/simple #已更新至2.2.3
python -m pip install  -r requirements.txt  \
   -i https://mirror.baidu.com/pypi/simple

python -m pip list |grep opencv

rm -rf models_list
rm -rf models_list_run
rm -rf models_list_all
rm -rf models_list_rec

# dir
log_path=log
phases='train eval infer export_model model_clip predict'
for phase in $phases
do
if [[ -d ${log_path}/${phase} ]]; then
   echo -e "\033[33m ${log_path}/${phase} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${phase}
   echo -e "\033[33m ${log_path}/${phase} is created successfully!\033[0m"
fi
done

#找到diff yaml  &  拆分任务  &  定义要跑的model list
if [[ ${model_flag} =~ 'CE' ]] || [[ ${model_flag} =~ 'CI_step1' ]] || [[ ${model_flag} =~ 'CI_step2' ]] \
    || [[ ${model_flag} =~ 'CI_step0' ]] \
    || [[ ${model_flag} =~ 'all' ]] || [[ ${model_flag} =~ 'pr' ]] || [[ ${model_flag} =~ 'clas' ]]; then
    find ppcls/configs/ImageNet/ -name '*.yaml' -exec ls -l {} \; \
        | awk '{print $NF;}'| grep -v 'eval' | grep -v 'kunlun' |grep -v 'distill' \
        |grep -v 'ResNeXt101_32x48d_wsl' |grep -v 'ResNeSt101' \
        > models_list_all #ResNeXt101_32x48d_wsl OOM   fp16 seresnet存在问题

    if [[ ${model_flag} =~ 'CI_step0' ]]; then
        cat models_list_all | while read line
        do
        if [[ ${line} =~ 'AlexNet' ]] ||[[ ${line} =~ 'DPN' ]] ||[[ ${line} =~ 'DarkNet' ]] ||[[ ${line} =~ 'DeiT' ]] \
            ||[[ ${line} =~ 'DenseNet' ]] ||[[ ${line} =~ 'EfficientNet' ]] ||[[ ${line} =~ 'GhostNet' ]] \
            ||[[ ${line} =~ 'HRNet' ]] ||[[ ${line} =~ 'HarDNet' ]] ||[[ ${line} =~ 'Inception' ]] \
            ||[[ ${line} =~ 'LeViT' ]] ||[[ ${line} =~ 'MixNet' ]] ||[[ ${line} =~ 'MobileNetV1' ]] \
            ||[[ ${line} =~ 'MobileNetV2' ]] ||[[ ${line} =~ 'MobileNetV3' ]]; then
            echo ${line}  >> models_list
        fi
        done

    elif [[ ${model_flag} =~ 'CI_step1' ]]; then
        cat models_list_all | while read line
        do
        if [[ ${line} =~ 'PPLCNet' ]] || [[ ${line} =~ 'ReXNet' ]] ||[[ ${line} =~ 'RedNet' ]] \
            ||[[ ${line} =~ 'Res2Net' ]] ||[[ ${line} =~ 'ResNeSt' ]] ||[[ ${line} =~ 'ResNeXt' ]]\
            ||[[ ${line} =~ 'ResNeXt101_wsl' ]] ||[[ ${line} =~ 'ResNet' ]] ||[[ ${line} =~ 'SENet' ]]\
            ||[[ ${line} =~ 'ShuffleNet' ]] ||[[ ${line} =~ 'ShuffleNet' ]] ||[[ ${line} =~ 'SqueezeNet' ]]; then
            echo ${line}  >> models_list
        fi
        done

    elif [[ ${model_flag} =~ "pr" ]];then
        shuf -n ${pr_num} models_list_all > models_list

    elif [[ ${model_flag} =~ "clas_single" ]] || [[ ${model_flag} =~ "CE" ]]; then
        echo $1 > models_list

    elif [[ ${model_flag} =~ 'CI_step2' ]]; then
        cat models_list_all | while read line
        do
        if [[ ! ${line} =~ 'AlexNet' ]] && [[ ! ${line} =~ 'DPN' ]] && [[ ! ${line} =~ 'DarkNet' ]] \
            && [[ ! ${line} =~ 'DeiT' ]] && [[ ! ${line} =~ 'DenseNet' ]] && [[ ! ${line} =~ 'EfficientNet' ]] \
            && [[ ! ${line} =~ 'GhostNet' ]] && [[ ! ${line} =~ 'HRNet' ]] && [[ ! ${line} =~ 'HarDNet' ]] \
            && [[ ! ${line} =~ 'Inception' ]] && [[ ! ${line} =~ 'LeViT' ]] && [[ ! ${line} =~ 'MixNet' ]] \
            && [[ ! ${line} =~ 'MobileNetV1' ]] && [[ ! ${line} =~ 'MobileNetV2' ]] && [[ ! ${line} =~ 'MobileNetV3' ]] \
            && [[ ! ${line} =~ 'PPLCNet' ]] && [[ ! ${line} =~ 'ReXNet' ]] && [[ ! ${line} =~ 'RedNet' ]] \
            && [[ ! ${line} =~ 'Res2Net' ]] && [[ ! ${line} =~ 'ResNeSt' ]] && [[ ! ${line} =~ 'ResNeXt' ]]\
            && [[ ! ${line} =~ 'ResNeXt101_wsl' ]] && [[ ! ${line} =~ 'ResNet' ]] && [[ ! ${line} =~ 'SENet' ]]\
            && [[ ! ${line} =~ 'ShuffleNet' ]] && [[ ! ${line} =~ 'SqueezeNet' ]]; then
            echo ${line}  >> models_list
        fi
        done

    elif [[ ${model_flag} =~ 'CI_all' ]] ; then #对应CI_ALL的情况
        shuf models_list_all > models_list

    fi

    echo "######  length models_list"
    wc -l models_list
    cat models_list
    if [[ ${model_flag} =~ "pr" ]];then
        # git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            # | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep ImageNet|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
        echo "######  diff models_list_diff"
        wc -l models_list_diff
        cat models_list_diff
        shuf -n 5 models_list_diff >> models_list #防止diff yaml文件过多导致pr时间过长

        if [[ ${static_flag} =~ "on" ]];then
            #增加静态图验证 只跑一个不放在循环中
            python -m paddle.distributed.launch ppcls/static/train.py  \
                -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
                -o Global.epochs=1 \
                -o DataLoader.Train.sampler.batch_size=1 \
                -o DataLoader.Eval.sampler.batch_size=1  \
                -o Global.output_dir=output \
                > $log_path/train/ResNet50_static.log 2>&1
            params_dir=$(ls output)
            echo "######  params_dir"
            echo $params_dir
            cat $log_path/train/ResNet50_static.log | grep "Memory Usage (MB)"

            if ([[ -f "output/$params_dir/latest.pdparams" ]] || [[ -f "output/$params_dir/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
                && [[ $(grep -c  "Error" $log_path/train/ResNet50_static.log) -eq 0 ]];then
                echo -e "\033[33m training static multi of ResNet50  successfully!\033[0m"|tee -a $log_path/result.log
                echo "training_static_exit_code: 0.0" >> $log_path/train/ResNet50_static.log
            else
                cat $log_path/train/ResNet50_static.log
                echo -e "\033[31m training static multi of ResNet50 failed!\033[0m"|tee -a $log_path/result.log
                echo "training_static_exit_code: 1.0" >> $log_path/train/ResNet50_static.log
            fi
        fi

    fi
    cat models_list | sort | uniq > models_list_run_tmp  #去重复
    shuf models_list_run_tmp > models_list_run
    rm -rf models_list_run_tmp
    echo "######  run models_list"
    wc -l models_list_run
    cat models_list_run

    #开始循环
    cat models_list_run | while read line
    do
    #echo $line
    filename=${line##*/}
    #echo $filename
    model=${filename%.*}
    echo $model

    if [[ ${line} =~ 'fp16' ]] || [[ ${line} =~ 'amp' ]];then
        echo "fp16 or amp"
        python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
        --upgrade nvidia-dali-cuda102 --ignore-installed -i https://mirror.baidu.com/pypi/simple
    fi

    if [ -d "output" ]; then
        rm -rf output
    fi

    #train
    #多卡
    if [[ ${model_flag} =~ "CE" ]]; then
        if [[ ${line} =~ 'GoogLeNet' ]] || [[ ${line} =~ 'VGG' ]] || [[ ${line} =~ 'ViT' ]] \
            || [[ ${line} =~ 'PPLCNet' ]] || [[ ${line} =~ 'MobileNetV3' ]]; then
            sed -i 's/learning_rate:/learning_rate: 0.0001 #/g' $line #将 学习率调低为0.0001
            echo "change lr"
        fi
        sed -i 's/RandCropImage/ResizeImage/g' $line
        sed -ie '/RandFlipImage/d' $line
        sed -ie '/flip_code/d' $line
        # -o Global.eval_during_train=False  \
        python -m paddle.distributed.launch tools/train.py -c $line  \
            -o Global.epochs=5  \
            -o Global.seed=1234 \
            -o Global.output_dir=output \
            -o DataLoader.Train.loader.num_workers=0 \
            -o DataLoader.Train.sampler.shuffle=False  \
            -o Global.eval_interval=5  \
            -o Global.save_interval=5 \
            -o DataLoader.Train.sampler.batch_size=4 \
            > $log_path/train/${model}_2card.log 2>&1
    else
        if [[ ! ${line} =~ "fp16.yaml" ]]; then
            if [[ `cat ${line} |grep MultiScaleSampler|wc -l` -gt "0"  ]]; then #for tingquan 220513 change
                echo "have MultiScaleSampler"
                python -m paddle.distributed.launch tools/train.py  \
                    -c $line -o Global.epochs=1 \
                    -o Global.output_dir=output \
                    -o DataLoader.Train.sampler.first_bs=1 \
                    -o DataLoader.Eval.sampler.batch_size=1  \
                    > $log_path/train/${model}_2card.log 2>&1
            else
                python -m paddle.distributed.launch tools/train.py  \
                    -c $line -o Global.epochs=1 \
                    -o Global.output_dir=output \
                    -o DataLoader.Train.sampler.batch_size=1 \
                    -o DataLoader.Eval.sampler.batch_size=1  \
                    > $log_path/train/${model}_2card.log 2>&1
            fi
        else
            python -m paddle.distributed.launch ppcls/static/train.py  \
                -c $line -o Global.epochs=1 \
                -o Global.output_dir=output \
                -o DataLoader.Train.sampler.batch_size=1 \
                -o DataLoader.Eval.sampler.batch_size=1  \
                > $log_path/train/${model}_2card.log 2>&1
        fi
    fi
    params_dir=$(ls output)
    echo "######  params_dir"
    echo $params_dir
    cat $log_path/train/${model}_2card.log | grep "Memory Usage (MB)"

    if ([[ -f "output/$params_dir/latest.pdparams" ]] || [[ -f "output/$params_dir/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
        && [[ $(grep -c  "Error" $log_path/train/${model}_2card.log) -eq 0 ]];then
        echo -e "\033[33m training multi of $model  successfully!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 0.0" >> $log_path/train/${model}_2card.log
    else
        cat $log_path/train/${model}_2card.log
        echo -e "\033[31m training multi of $model failed!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 1.0" >> $log_path/train/${model}_2card.log
    fi

    if [[ ${line} =~ "fp16.yaml" ]]; then #无单独fp16的yaml了
        continue
    fi

    #单卡
    ls output/$params_dir/ |head -n 2
    sleep 3
    if [[ ${model_flag} =~ "CE" ]]; then
        rm -rf output #清空多卡cache
        python  tools/train.py -c $line  \
            -o Global.epochs=5  \
            -o Global.seed=1234 \
            -o Global.output_dir=output \
            -o DataLoader.Train.loader.num_workers=0 \
            -o DataLoader.Train.sampler.shuffle=False  \
            -o Global.eval_interval=5  \
            -o Global.save_interval=5 \
            -o DataLoader.Train.sampler.batch_size=4  \
            > $log_path/train/${model}_1card.log 2>&1
    # else  #取消CI单卡训练
    #    python tools/train.py  \
    #       -c $line -o Global.epochs=1 \
    #       -o Global.output_dir=output \
    #       -o DataLoader.Train.sampler.batch_size=1 \
    #       -o DataLoader.Eval.sampler.batch_size=1  \
    #       > $log_path/train/${model}_1card.log 2>&1
        params_dir=$(ls output)
        echo "######  params_dir"
        echo $params_dir
        cat $log_path/train/${model}_1card.log | grep "Memory Usage (MB)"
        if [[ -f "output/$params_dir/latest.pdparams" ]] && [[ $? -eq 0 ]] \
            && [[ $(grep -c  "Error" $log_path/train/${model}_1card.log) -eq 0 ]];then
            echo -e "\033[33m training single of $model  successfully!\033[0m"|tee -a $log_path/result.log
            echo "training_single_exit_code: 0.0" >> $log_path/train/${model}_1card.log
        else
            cat $log_path/train/${model}_1card.log
            echo -e "\033[31m training single of $model failed!\033[0m"|tee -a $log_path/result.log
            echo "training_single_exit_code: 1.0" >> $log_path/train/${model}_1card.log
        fi
    fi

    if  [[ ${model} =~ 'RedNet' ]] || [[ ${line} =~ 'LeViT' ]] || [[ ${line} =~ 'GhostNet' ]] \
        || [[ ${line} =~ 'ResNet152' ]] || [[ ${line} =~ 'DLA169' ]] || [[ ${line} =~ 'ResNeSt101' ]] \
        || [[ ${line} =~ 'ResNeXt152_vd_64x4d' ]] || [[ ${line} =~ 'ResNeXt152_64x4d' ]] || [[ ${line} =~ 'ResNet101' ]] \
        || [[ ${line} =~ 'ResNet200_vd' ]] || [[ ${line} =~ 'DLA102' ]] || [[ ${model} =~ 'InceptionV4' ]];then
        echo "######  use pretrain model"
        echo ${model}
        wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/${model}_pretrained.pdparams --no-proxy
        rm -rf output/$params_dir/latest.pdparams
        cp -r ${model}_pretrained.pdparams output/$params_dir/latest.pdparams
        rm -rf ${model}_pretrained.pdparams
    fi

    if [[ ${model} =~ 'MobileNetV3' ]] || ( [[ ${model} =~ 'PPLCNet' ]] && [[ ! ${model} =~ 'dml' ]] ) \
        || [[ ${line} =~ 'ESNet' ]] || [[ ${line} =~ 'ResNet50.yaml' ]] || [[ ${line} =~ '/ResNet50_vd.yaml' ]];then
        echo "######  use pretrain model"
        echo ${model}
        wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/${model}_pretrained.pdparams --no-proxy
        rm -rf output/$params_dir/latest.pdparams
        cp -r ${model}_pretrained.pdparams output/$params_dir/latest.pdparams
        rm -rf ${model}_pretrained.pdparams
    fi

    if [[ ${model} =~ 'amp' ]];then
        echo "######  use amp pretrain model"
        echo ${model}
        wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/${params_dir}_pretrained.pdparams --no-proxy
        rm -rf output/$params_dir/latest.pdparams
        cp -r ${params_dir}_pretrained.pdparams output/$params_dir/latest.pdparams
        rm -rf ${params_dir}_pretrained.pdparams
    fi

    if [[ ${model} =~ 'distill_pphgnet_base' ]]  || [[ ${model} =~ 'PPHGNet_base' ]] ;then
        echo "######  use distill_pphgnet_base pretrain model"
        echo ${model}
        echo ${params_dir}
        wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_base_ssld_pretrained.pdparams --no-proxy
        rm -rf output/$params_dir/latest.pdparams
        cp -r PPHGNet_base_ssld_pretrained.pdparams output/$params_dir/latest.pdparams
        rm -rf PPHGNet_base_ssld_pretrained_pretrained.pdparams
    fi

    if [[ ${model} =~ 'PPLCNet' ]]  && [[ ${model} =~ 'dml' ]] ;then
        echo "######  use PPLCNet dml pretrain model"
        echo ${model}
        echo ${params_dir}
        wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Distillation/${model}_pretrained.pdparams --no-proxy
        rm -rf output/$params_dir/latest.pdparams
        cp -r ${model}_pretrained.pdparams output/$params_dir/latest.pdparams
        rm -rf ${model}_pretrained.pdparams
    fi

    sleep 3

    ls output/$params_dir/ |head -n 2
    # eval
    if [[ ${line} =~ 'ultra' ]];then
        cp ${line} ${line}_tmp #220413 fix tingquan
        sed -i '/output_fp16: True/d' ${line}
    fi

    python tools/eval.py -c $line \
        -o Global.pretrained_model=output/$params_dir/latest \
        -o DataLoader.Eval.sampler.batch_size=1 \
        > $log_path/eval/$model.log 2>&1

    if [[ ${line} =~ 'ultra' ]];then
        rm -rf ${line}
        mv ${line}_tmp ${line} #220413 fix tingquan
    fi

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/eval/${model}.log) -eq 0 ]];then
        echo -e "\033[33m eval of $model  successfully!\033[0m"| tee -a $log_path/result.log
        echo "eval_exit_code: 0.0" >> $log_path/eval/$model.log
    else
        cat $log_path/eval/$model.log
        echo -e "\033[31m eval of $model failed!\033[0m" | tee -a $log_path/result.log
        echo "eval_exit_code: 1.0" >> $log_path/eval/$model.log
    fi

    # infer
    python tools/infer.py -c $line \
        -o Global.pretrained_model=output/$params_dir/latest \
        > $log_path/infer/$model.log 2>&1
    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/infer/${model}.log) -eq 0 ]];then
        echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
        echo "infer_exit_code: 0.0" >> $log_path/infer/$model.log
    else
        cat $log_path/infer/${model}.log
        echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
        echo "infer_exit_code: 1.0" >> $log_path/infer/$model.log
    fi

    # export_model
    if [[ ${line} =~ 'fp16' ]] || [[ ${line} =~ 'amp' ]];then
        python tools/export_model.py -c $line \
            -o Global.pretrained_model=output/$params_dir/latest \
            -o Global.save_inference_dir=./inference/$model \
            -o Arch.data_format="NCHW" \
            > $log_path/export_model/$model.log 2>&1
    else
        python tools/export_model.py -c $line \
            -o Global.pretrained_model=output/$params_dir/latest \
            -o Global.save_inference_dir=./inference/$model \
            > $log_path/export_model/$model.log 2>&1
    fi

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/export_model/${model}.log) -eq 0 ]];then
        echo -e "\033[33m export_model of $model  successfully!\033[0m"| tee -a $log_path/result.log
        echo "export_exit_code: 0.0" >> $log_path/export_model/$model.log
    else
        cat $log_path/export_model/$model.log
        echo -e "\033[31m export_model of $model failed!\033[0m" | tee -a $log_path/result.log
        echo "export_exit_code: 1.0" >> $log_path/export_model/$model.log
    fi

    # 20220325 error
    # if [[ `expr $RANDOM % 2` -eq 0 ]] && ([[ ${model_flag} =~ 'CI' ]] || [[ ${model_flag} =~ 'single' ]]);then
    # # if [[ ${model_flag} =~ 'CI' ]] || [[ ${model_flag} =~ 'single' ]];then #加入随机扰动
    #    echo "model_clip"
    #    python model_clip.py --path_prefix="./inference/$model/inference" \
    #       --output_model_path="./inference/$model/inference" \
    #       > $log_path/model_clip/$model.log 2>&1
    #    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/model_clip/${model}.log) -eq 0 ]];then
    #       echo -e "\033[33m model_clip of $model  successfully!\033[0m"| tee -a $log_path/result.log
    #    else
    #       cat $log_path/model_clip/$model.log
    #       echo -e "\033[31m model_clip of $model failed!\033[0m" | tee -a $log_path/result.log
    #    fi
    # fi

    size_tmp=`cat ${line} |grep image_shape|cut -d "," -f2|cut -d " " -f2` #获取train的shape保持和predict一致
    cd deploy
    # sed -i 's/size: 224/size: 384/g' configs/inference_cls.yaml
    sed -i 's/size: 224/size: '${size_tmp}'/g' configs/inference_cls.yaml #修改predict尺寸
    # sed -i 's/resize_short: 256/resize_short: 384/g' configs/inference_cls.yaml
    sed -i 's/resize_short: 256/resize_short: '${size_tmp}'/g' configs/inference_cls.yaml

    if [[ ${line} =~ 'fp16' ]] || [[ ${line} =~ 'ultra' ]];then
        python python/predict_cls.py -c configs/inference_cls_ch4.yaml \
            -o Global.inference_model_dir="../inference/"$model \
            > ../$log_path/predict/$model.log 2>&1
    else
        python python/predict_cls.py -c configs/inference_cls.yaml \
            -o Global.inference_model_dir="../inference/"$model \
            > ../$log_path/predict/$model.log 2>&1
    fi

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ../$log_path/predict/${model}.log) -eq 0 ]];then
        echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
    else
        cat ../$log_path/predict/${model}.log
        echo -e "\033[31m predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
    fi

    if [[ ${line} =~ 'fp16' ]] || [[ ${line} =~ 'ultra' ]];then
        python python/predict_cls.py -c configs/inference_cls_ch4.yaml  \
            -o Global.infer_imgs="./images"  \
            -o Global.batch_size=4 -o Global.inference_model_dir="../inference/"$model \
            > ../$log_path/predict/$model.log 2>&1
    else
        python python/predict_cls.py -c configs/inference_cls.yaml  \
            -o Global.infer_imgs="./images"  \
            -o Global.batch_size=4 \
            -o Global.inference_model_dir="../inference/"$model \
            > ../$log_path/predict/$model.log 2>&1
    fi
    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ../$log_path/predict/${model}.log) -eq 0 ]];then
        echo -e "\033[33m multi_batch_size predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
        echo "predict_exit_code: 0.0" >> ../$log_path/predict/$model.log
    else
        cat ../$log_path/predict/${model}.log
        echo -e "\033[31m multi_batch_size predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
        echo "predict_exit_code: 1.0" >> ../$log_path/predict/$model.log
    fi

    sed -i 's/size: '${size_tmp}'/size: 224/g' configs/inference_cls.yaml #改回predict尺寸
    sed -i 's/resize_short: '${size_tmp}'/resize_short: 256/g' configs/inference_cls.yaml

    cd ..
    done
fi

if [[ ${model_flag} =~ 'CI_step3' ]] || [[ ${model_flag} =~ 'all' ]] || [[ ${model_flag} =~ 'pr' ]] || [[ ${model_flag} =~ 'rec' ]]; then
    echo "######  rec step"
    rm -rf models_list
    rm -rf models_list_run

    find ppcls/configs/Cartoonface/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list_rec
    find ppcls/configs/Logo/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list_rec
    find ppcls/configs/Products/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec
    find ppcls/configs/Vehicle/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec
    find ppcls/configs/slim/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec #后续改成slim
    find ppcls/configs/GeneralRecognition/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' \
        |grep -v 'Gallery2FC_PPLCNet_x2_5' >> models_list_rec #后续改成slim
    find ppcls/configs/DeepHash/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec #后续改成deephash

    if [[ ${model_flag} =~ 'pr' ]]; then
        shuf -n ${pr_num} models_list_rec > models_list
    elif [[ ${model_flag} =~ "rec_single" ]];then
        echo $1 > models_list
    elif [[ ${model_flag} =~ "CI" ]];then
        shuf models_list_rec > models_list
    fi
    echo "######  rec models_list"
    wc -l models_list
    cat models_list

    if [[ ${model_flag} =~ "pr" ]];then
        # git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            # | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep Cartoonface|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep Logo|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep Products|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep Vehicle|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep slim|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep GeneralRecognition|awk -F 'b/' '{print$2}' |tee -a  models_list_diff_rec
        git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
            | grep diff|grep yaml|grep configs|grep DeepHash|awk -F 'b/' '{print$2}' |tee -a  models_list_diff_rec
        shuf -n 3 models_list_diff_rec >> models_list #防止diff yaml文件过多导致pr时间过长
        echo "######  diff models_list_diff"
        wc -l models_list_diff_rec
        cat models_list_diff_rec
    fi
    echo "######  rec run models_list"
    wc -l models_list
    cat models_list

    #train
    cat models_list | while read line
    do

    if [[ ${line} =~ 'Gallery2FC' ]]; then
        echo "have Gallery2FC"
        continue
    fi

    #echo $line
    filename=${line##*/}
    #echo $filename
    model=${filename%.*}
    category=`echo $line | awk -F [\/] '{print $3}'`
    echo ${category}_${model}
    echo $category

    if [ -d "output" ]; then
        rm -rf output
    fi
    echo $model

    # sleep 3

    if [[ ${line} =~ 'GeneralRecognition' ]]; then
        python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Train.sampler.batch_size=32 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    elif [[ ${line} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
        python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Eval.sampler.batch_size=64 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    elif [[ ${line} =~ 'quantization' ]] ; then
        python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Train.sampler.batch_size=32 \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    else
        python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    fi

    params_dir=$(ls output/${category}_${model})
    echo "######  params_dir"
    echo $params_dir
    cat $log_path/train/${category}_${model}.log | grep "Memory Usage (MB)"

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/train/${category}_${model}.log) -eq 0 ]] \
        && [[ -f "output/${category}_${model}/$params_dir/latest.pdparams" ]];then
        echo -e "\033[33m training of ${category}_${model}  successfully!\033[0m"|tee -a $log_path/result.log
    else
        cat $log_path/train/${category}_${model}.log
        echo -e "\033[31m training of ${category}_${model} failed!\033[0m"|tee -a $log_path/result.log
    fi

    # sleep 3

    # eval
    if [[ ${line} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
        python tools/eval.py -c $line \
            -o Global.pretrained_model=output/${category}_${model}/$params_dir/latest \
            -o DataLoader.Eval.sampler.batch_size=64 \
            > $log_path/eval/${category}_${model}.log 2>&1
    else
        python tools/eval.py -c $line \
            -o Global.pretrained_model=output/${category}_${model}/$params_dir/latest \
            > $log_path/eval/${category}_${model}.log 2>&1
    fi
    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/eval/${category}_${model}.log) -eq 0 ]];then
        echo -e "\033[33m eval of ${category}_${model}  successfully!\033[0m"| tee -a $log_path/result.log
    else
        cat $log_path/eval/${category}_${model}.log
        echo -e "\033[31m eval of ${category}_${model} failed!\033[0m" | tee -a $log_path/result.log
    fi

    # export_model
    python tools/export_model.py -c $line \
        -o Global.pretrained_model=output/${category}_${model}/$params_dir/latest  \
        -o Global.save_inference_dir=./inference/${category}_${model} \
        > $log_path/export_model/${category}_${model}.log 2>&1
    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/export_model/${category}_${model}.log) -eq 0 ]];then
        echo -e "\033[33m export_model of ${category}_${model}  successfully!\033[0m"| tee -a $log_path/result.log
    else
        cat $log_path/export_model/${category}_${model}.log
        echo -e "\033[31m export_model of ${category}_${model} failed!\033[0m" | tee -a $log_path/result.log
    fi

    # predict
    cd deploy
    case $category in
        Cartoonface)
        python  python/predict_system.py -c configs/inference_cartoon.yaml \
            > ../$log_path/predict/cartoon.log 2>&1
        if [ $? -eq 0 ];then
            echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
        else
            cat ../$log_path/predict/cartoon.log
        echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
        fi
    ;;
    Logo)
        python  python/predict_system.py -c configs/inference_logo.yaml \
            > ../$log_path/predict/logo.log 2>&1
        if [ $? -eq 0 ];then
            echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
        else
            cat ../$log_path/predict/logo.log
        echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
        fi
    ;;
    Products)
        python  python/predict_system.py -c configs/inference_product.yaml \
            > ../$log_path/predict/product.log 2>&1
        if [ $? -eq 0 ];then
            echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
        else
            cat ../$log_path/predict/product.log
        echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
        fi
    ;;
    Vehicle)
        python  python/predict_system.py -c configs/inference_vehicle.yaml \
            > ../$log_path/predict/vehicle.log 2>&1
        if [ $? -eq 0 ];then
            echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
        else
            cat ../$log_path/predict/vehicle.log
        echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
        fi
    ;;
    esac
    cd ..

    done
fi

num=`cat $log_path/result.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
    if [[ ! ${model_flag} =~ 'CE' ]]; then
        echo -e "-----------------------------base cases-----------------------------"
        cat $log_path/result.log | grep "failed"
        echo -e "--------------------------------------------------------------------"
    fi
    exit 1
    else
    exit 0
fi
