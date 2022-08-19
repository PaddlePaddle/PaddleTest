#定义环境变量
export FLAGS_cudnn_deterministic=True #固定随机量使用，使cuda算法保持一致

# paddle
echo "######  paddle version"
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

# repo
echo "######  clas version"
git rev-parse HEAD

# python
python -c 'import sys; print(sys.version_info[:])'
echo "######  python version"

# system
if [ -d "/etc/redhat-release" ]; then
    echo "######  system centos"
else
    echo "######  system linux"
fi

#安装向上取整依赖包，需要代理
yum install bc -y >/dev/null 2>&1
apt-get install bc -y >/dev/null 2>&1

#取消代理用镜像安装包，安装依赖包
unset http_proxy
unset https_proxy

python -m pip install --upgrade \
    pip -i https://mirror.baidu.com/pypi/simple  >/dev/null 2>&1
# python -m pip install -U pyyaml \
#     -i https://mirror.baidu.com/pypi/simple  >/dev/null 2>&1
python -m pip install -U paddleslim \
    -i https://mirror.baidu.com/pypi/simple  >/dev/null 2>&1
python -m pip install  -r requirements.txt  \
    -i https://mirror.baidu.com/pypi/simple  >/dev/null 2>&1

if [[ ${yaml_line} =~ "face" ]] && [[ ${yaml_line} =~ "metric_learning" ]];then
    echo "metric_learning face"
    # 更新 pip/setuptools
    python -m  pip install -U pip setuptools cython \
        -i https://mirror.baidu.com/pypi/simple  >/dev/null 2>&1
    # 安装 bcolz
    python -m  pip install bcolz==1.2.0  \
        -i https://mirror.baidu.com/pypi/simple  >/dev/null 2>&1
fi
if [[ ${yaml_line} =~ 'fp16' ]] || [[ ${yaml_line} =~ 'amp' ]];then
    echo "fp16 or amp"
    # python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    # --upgrade nvidia-dali-cuda102 --ignore-installed -i https://mirror.baidu.com/pypi/simple
    if [[ -f "nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl" ]] && \
        [[ -f "nvidia_dali_cuda110-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl" ]] ;then
        echo "already download nvidia_dali_cuda102 nvidia_dali_cuda110"
    else
        wget -q https://paddle-qa.bj.bcebos.com/PaddleClas/nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl --no-proxy
        wget -q https://paddle-qa.bj.bcebos.com/PaddleClas/nvidia_dali_cuda110-1.8.0-3362434-py3-none-manylinux2014_x86_64.whl --no-proxy
    fi
    python -m pip install nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl
    python -m pip install nvidia_dali_cuda110-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl

    export FLAGS_cudnn_deterministic=False #amp单独考虑，不能固定随机量，否则报错如下
    # InvalidArgumentError: Cann't set exhaustive_search True and FLAGS_cudnn_deterministic True at same time.
fi
python setup.py install >/dev/null 2>&1 #安装whl包

#确定log存储位置
export log_path=../log
export output_dir=output
phases='train eval infer export_model predict api_test'
for phase in $phases
do
if [[ ! -d ${log_path}/${phase} ]]; then
   mkdir -p  ${log_path}/${phase}
   echo -e "\033[33m ${log_path}/${phase} is created successfully!\033[0m"
fi
done

# 获取模型名称
# array=(${line//\// })
array=(${yaml_line//\// })
export model_type=${array[2]} #区分 分类、slim、识别等
export model_name=${array[2]} #进行字符串拼接
if [[ ${yaml_line} =~ "PULC" ]];then
    export model_type_PULC=${array[3]} #PULC为了区分9中类别单独区分
fi
echo "### model_type"
echo ${model_type}
for var in ${array[@]:3}
do
    array2=(${var//'.yaml'/ })
    export model_name=${model_name}-${array2[0]}
done
export model_latest_name=${array2[0]}
echo "### model_latest_name"
echo ${model_latest_name}
echo "### model_name"
echo ${model_name}

#获取模型输出名称、评估下载名称、预测下载名称
# import yaml
# params_dir=`python -c "
#     import yaml; \
#     with open('${yaml_line}', 'r', encoding='utf-8') as y:; \
#     cfg = yaml.full_load(y); \
#     print(cfg['Arch']['name']); \
#     "`
function get_params(){
    params_index=(`cat ${yaml_line} | grep -n $1 | awk -F ":" '{print $1}'`)
    params_word=`sed -n "${params_index[0]},$[${params_index[0]}+3]p" ${yaml_line}`
    params_dir=(`echo ${params_word} | grep name: | awk -F ":" '{print $3}'`)
    params_dir=(${params_dir//\"/ })
    params_dir=(${params_dir//\"/ })
    echo ${params_dir}
}
export params_dir=`get_params Arch:`
echo "#### params_dir"
echo ${params_dir}
if [[ ${params_dir} == "RecModel" ]];then
    if [[ `cat ${yaml_line}` =~ "Backbone" ]];then
        pdparams_pretrain=`get_params Backbone:`
    else
        pdparams_pretrain=`get_params Arch:`
    fi
elif [[ ${params_dir} == "DistillationModel" ]];then
    if [[ `cat ${yaml_line}` =~ "Backbone" ]];then
        pdparams_pretrain=`get_params Backbone:`
    else
        pdparams_pretrain=`get_params Student:`
    fi
else
    pdparams_pretrain=${params_dir}
fi
export pdparams_pretrain=(${pdparams_pretrain//"_Tanh"/ })
export pdparams_pretrain=(${pdparams_pretrain//"_last_stage_stride1"/ })
if [[ ${pdparams_pretrain} == "AttentionModel" ]];then
    export pdparams_pretrain="ResNet18"
fi
if [[ ${model_type} == "PULC" ]];then
    export infer_pretrain=${model_type_PULC}
else
    export infer_pretrain=${pdparams_pretrain}
fi
echo "#### pdparams_pretrain"
echo ${pdparams_pretrain}
echo "#### infer_pretrain"
echo ${infer_pretrain}


#对32G的模型进行bs减半的操作，注意向上取整 #暂时适配了linux，未考虑MAC
# if [[ 'ImageNet_CSPNet_CSPDarkNet53 ImageNet_DPN_DPN107 ImageNet_DeiT_DeiT_tiny_patch16_224 \
#     ImageNet_EfficientNet_EfficientNetB0 ImageNet_GhostNet_GhostNet_x1_3 ImageNet_RedNet_RedNet50 \
#     ImageNet_ResNeXt101_wsl_ResNeXt101_32x8d_wsl ImageNet_ResNeXt_ResNeXt152_64x4d \
#     ImageNet_SwinTransformer_SwinTransformer_tiny_patch4_window7_224 ImageNet_TNT_TNT_small \
#     ImageNet_Twins_alt_gvt_small ImageNet_Twins_pcpvt_small ImageNet_Xception_Xception41_deeplab \
#     ImageNet_Xception_Xception71' =~ ${model_name} ]];then
#     echo "change ${model_name} batch_size"
#     yum install bc -y
#     apt-get install bc -y
#     function ceil(){
#     floor=`echo "scale=0;$1/1"|bc -l ` # 向上取整 局部变量$1不影响
#     add=`awk -v num1=$floor -v num2=$1 'BEGIN{print(num1<num2)?"1":"0"}'`
#     echo `expr $floor  + $add`
#     }
#     index=(`cat ${yaml_line} | grep -n batch_size | awk -F ":" '{print $1}'`)
#     for((i=0;i<${#index[@]};i++));
#     do
#         num_str=`sed -n ${index[i]}p ${yaml_line}`
#         if [[ ${num_str} =~ "#@" ]];then #  #@ 保证符号的唯一性
#             continue
#         fi
#         input_num=(`echo ${num_str} | grep -o -E '[0-9]+'  | sed -e 's/^0\+//'`)
#         ((Div=${input_num[0]} %2))
#         if [ "${Div}" == 0 ];then
#             out_num=`expr ${input_num[0]}/2 |bc` #整除2
#         else
#             echo "can not %2 will ceil"
#             out_num=`expr ${input_num[0]}/2` #bs向上取整
#             out_num=`ceil ${out_num}`
#         fi
#         sed -i "${index[i]}s/batch_size: /batch_size: ${out_num} #@/" ${yaml_line}
#     done
# fi

#默认bath_size除以3
if [[ ${model_name} =~ "reid-strong_baseline" ]] || [[ ${model_name} =~ "Logo-ResNet50_ReID" ]];then
    echo "do no need change bs" #针对特殊的sampler 需要满足整除某数的bs
else
    function ceil(){
    floor=`echo "scale=0;$1/1"|bc -l ` # 向上取整 局部变量$1不影响
    add=`awk -v num1=$floor -v num2=$1 'BEGIN{print(num1<num2)?"1":"0"}'`
    echo `expr $floor  + $add`
    }
    index=(`cat ${yaml_line} | grep -n batch_size | awk -F ":" '{print $1}'`)
    for((i=0;i<${#index[@]};i++));
    do
        num_str=`sed -n ${index[i]}p ${yaml_line}`
        if [[ ${num_str} =~ "#@" ]];then #  #@ 保证符号的唯一性
            continue
        fi
        input_num=(`echo ${num_str} | grep -o -E '[0-9]+'  | sed -e 's/^0\+//'`)
        ((Div=${input_num[0]} %3))
        if [ "${Div}" == 0 ];then
            out_num=`expr ${input_num[0]}/3 |bc` #整除2
        else
            echo "can not %3 will ceil"
            out_num=`expr ${input_num[0]}/3` #bs向上取整
            out_num=`ceil ${out_num}`
        fi
        sed -i "${index[i]}s/batch_size: /batch_size: ${out_num} #@/" ${yaml_line}
        echo "change ${model_name} batch_size from ${input_num[0]} to ${out_num}"
    done
fi

#区分单卡多卡
# export CUDA_VISIBLE_DEVICES=  #这一步让框架来集成
if [[ ${cuda_type} =~ "SET_MULTI_CUDA" ]];then
    export card="2card"
    export Global_epochs="1"
    export multi_flag="-m paddle.distributed.launch"
    export set_cuda_device="gpu"
    export set_cuda_flag=True
elif [[ ${cuda_type} =~ "CPU" ]];then
    export card="cpu"
    export Global_epochs="1"
    export multi_flag=" "
    export set_cuda_device="cpu"
    export set_cuda_flag=Flase
else
    export card="1card"
    export Global_epochs="1"
    export multi_flag=" "
    export set_cuda_device="gpu"
    export set_cuda_flag=True
fi


#准备数据下载函数
get_image_name(){
    #传入split参数 image_root
    image_root_name=(`cat ${yaml_line} | grep  ${image_root} | awk -F ":" '{print $2}'`)
    image_root_name=(${image_root_name//dataset// })
    image_root_name=(${image_root_name[1]//\// })
    image_root_name=(${image_root_name//\"/ })
    export image_root_name=(${image_root_name//\"/ })
}

download_data(){
    #传入参数 image_root_name
    echo "download start image_root_name : ${image_root_name}"
    cd dataset #这里是默认按照已经进入repo路径来看
    if [[ -f "${image_root_name}.tar" ]] && [[ -d "${image_root_name}" ]] ;then
        echo already download ${image_root_name}
    else
wget -q -c https://paddle-qa.bj.bcebos.com/PaddleClas/ce_data/${image_root_name}.tar --no-proxy --no-check-certificate
        tar xf ${image_root_name}.tar
    fi
    echo "download done image_root_name : ${image_root_name}"
    cd ..
}

#准备数据
cd deploy
if [[ -f "recognition_demo_data_en_v1.1.tar" ]] && [[ -f "drink_dataset_v1.0.tar" ]] ;then
    echo already download rec_demo
else
wget -q -c https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar --no-proxy \
    && tar -xf drink_dataset_v1.0.tar
wget -q -c https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/recognition_demo_data_en_v1.1.tar \
    --no-proxy --no-check-certificate && tar xf recognition_demo_data_en_v1.1.tar
fi
cd ..

if [[ ${get_data_way} == "ln_way" ]];then
    if [[ ${Data_path} == "" ]];then
        echo " you must set Data_path first "
    fi
    echo "######  ----ln  data-----"
    rm -rf dataset
    ln -s ${Data_path} dataset
    ls dataset |head -n 2
else
    echo "######  ----download  data-----"
    if [[ ${yaml_line} =~ "face" ]] && [[ ${yaml_line} =~ "metric_learning" ]];then
        image_root="root_dir"
        get_image_name image_root
        download_data
    elif [[ ${yaml_line} =~ "traffic_sign" ]] && [[ ${yaml_line} =~ "PULC" ]];then
        image_root="cls_label_path"
        get_image_name image_root
        download_data
    elif [[ ${yaml_line} =~ "GeneralRecognition" ]];then
        export image_root_name="Inshop"
        download_data
        export image_root_name="Aliproduct"
        download_data
    elif [[ ${yaml_line} =~ "strong_baseline" ]] && [[ ${yaml_line} =~ "reid" ]];then
        export image_root_name="market1501"
        download_data
    elif [[ ${yaml_line} =~ "MV3_Large_1x_Aliproduct_DLBHC" ]] && [[ ${yaml_line} =~ "Products" ]];then
        image_root="image_root"
        get_image_name image_root
        download_data
        export image_root_name="Inshop"
        download_data
    else
        image_root="image_root"
        get_image_name image_root
        download_data
    fi
fi
####TODO，抽象出epoch数，CI 跑1个epoch，CE 跑2个epoch，控制下执行时间
