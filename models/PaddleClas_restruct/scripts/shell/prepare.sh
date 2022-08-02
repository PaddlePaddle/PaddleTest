#定义环境变量
export FLAGS_cudnn_deterministic=True #固定随机量使用，使cuda算法保持一致

echo "######  ----ln  data-----"
rm -rf dataset
ln -s ${Data_path} dataset
ls dataset |head -n 2
cd deploy
ln -s ${Data_path}/rec_demo/* .  #预训练模型和demo数据集
cd ..

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

#取消代理用镜像安装包
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
if [[ ${yaml_line} =~ 'fp16' ]] || [[ ${yaml_line} =~ 'amp' ]];then
    echo "fp16 or amp"
    python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    --upgrade nvidia-dali-cuda102 --ignore-installed -i https://mirror.baidu.com/pypi/simple
fi


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
echo ${model_type}
for var in ${array[@]:3}
do
    array2=(${var//'.yaml'/ })
    export model_name=${model_name}_${array2[0]}
done
export model_latest_name=${array2[0]}
echo ${model_latest_name}

#获取模型输出名
# import yaml
# params_dir=`python -c "
#     import yaml; \
#     with open('${yaml_line}', 'r', encoding='utf-8') as y:; \
#     cfg = yaml.full_load(y); \
#     print(cfg['Arch']['name']); \
#     "`
export params_dir=(`cat ${yaml_line} | grep name | awk -F ":" '{print $2}'`)
export params_dir=${params_dir//\"/ }
echo ${params_dir}

#对32G的模型进行bs减半的操作，注意向上取整 #暂时适配了linux，未考虑MAC
if [[ 'ImageNet_CSPNet_CSPDarkNet53 ImageNet_DPN_DPN107 ImageNet_DeiT_DeiT_tiny_patch16_224 \
    ImageNet_EfficientNet_EfficientNetB0 ImageNet_GhostNet_GhostNet_x1_3 ImageNet_RedNet_RedNet50 \
    ImageNet_ResNeXt101_wsl_ResNeXt101_32x8d_wsl ImageNet_ResNeXt_ResNeXt152_64x4d \
    ImageNet_SwinTransformer_SwinTransformer_tiny_patch4_window7_224 ImageNet_TNT_TNT_small \
    ImageNet_Twins_alt_gvt_small ImageNet_Twins_pcpvt_small ImageNet_Xception_Xception41_deeplab \
    ImageNet_Xception_Xception71' =~ ${model_name} ]];then
    echo "change ${model_name} batch_size"
    yum install bc -y
    apt-get install bc -y
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
        ((Div=${input_num[0]} %2))
        if [ "${Div}" == 0 ];then
            out_num=`expr ${input_num[0]}/2 |bc` #整除2
        else
            echo "can not %2 will ceil"
            out_num=`expr ${input_num[0]}/2` #bs向上取整
            out_num=`ceil ${out_num}`
        fi
        sed -i "${index[i]}s/batch_size: /batch_size: ${out_num} #@/" ${yaml_line}
    done
fi

#区分单卡多卡
# export CUDA_VISIBLE_DEVICES=  #这一步让框架来集成
if [[ ${cuda_type} =~ "SET_MULTI_CUDA" ]];then
    export card="2card"
    export multi_flag="-m paddle.distributed.launch"
    export set_cuda_device="gpu"
    export set_cuda_flag=True
elif [[ ${cuda_type} =~ "CPU" ]];then
    export card="cpu"
    export multi_flag=" "
    export set_cuda_device="cpu"
    export set_cuda_flag=Flase
else
    export card="1card"
    export multi_flag=" "
    export set_cuda_device="gpu"
    export set_cuda_flag=True
fi


####TODO，抽象出epoch数，CI 跑1个epoch，CE 跑2个epoch，控制下执行时间
