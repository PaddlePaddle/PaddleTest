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

python -m pip install --ignore-installed --upgrade \
    pip -i https://mirror.baidu.com/pypi/simple
python -m pip install  --ignore-installed paddleslim \
    -i https://mirror.baidu.com/pypi/simple
python -m pip install  -r requirements.txt  \
    -i https://mirror.baidu.com/pypi/simple


#确定log存储位置
export log_path=../log
export output_dir=output
phases='train eval infer export_model predict api_test'
for phase in $phases
do
if [[ -d ${log_path}/${phase} ]]; then
   echo -e "\033[33m ${log_path}/${phase} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${phase}
   echo -e "\033[33m ${log_path}/${phase} is created successfully!\033[0m"
fi
done

# 获取模型名称
# array=(${line//\// })
array=(${1//\// })
export model_type=${array[2]} #区分 分类、slim、识别等
export model_name=${array[2]} #进行字符串拼接
if [[ ${line} =~ "PULC" ]];then
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

#区分单卡多卡
# export CUDA_VISIBLE_DEVICES=  #这一步让框架来集成
if [[ ${2} =~ "SET_MULTI_CUDA" ]];then
    export card="2card"
    export multi_flag="-m paddle.distributed.launch"
    export set_cuda_device="gpu"
    export set_cuda_flag=True
elif [[ ${2} =~ "CPU" ]];then
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
