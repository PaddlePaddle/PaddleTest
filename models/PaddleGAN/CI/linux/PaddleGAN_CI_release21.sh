unset GREP_OPTIONS
echo ${cudaid1}
echo ${cudaid2}
echo ${Data_path}
echo ${paddle_compile}
export CUDA_VISIBLE_DEVICES=${cudaid2}
export FLAGS_use_virtual_memory_auto_growth=1
export FLAGS_use_stream_safe_cuda_allocator=1

#<-> model_flag CI是效率云  pr是TC，all是全量，single是单独模型debug
#<-> pr_num   随机跑pr的模型数
#<-> python   python版本
#<-> cudaid2   cuda卡号
#<-> paddle_compile   paddle包地址
#<-> Data_path   数据路径
#$1 <-> 自己定义single_yaml_debug  单独模型yaml字符


# data
echo "######  ----ln  data-----"
rm -rf data
ln -s ${Data_path} data
ls data

if [[ ${model_flag} =~ 'CI' ]]; then
   rm -rf /root/.cache/paddle/dataset/mnist
   if [ ! -f "/root/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz" ]; then
      wget -q -P /root/.cache/paddle/dataset/mnist/ https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz
   fi

   if [ ! -f "/root/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz" ]; then
      wget -q -P /root/.cache/paddle/dataset/mnist/ https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz
   fi
fi

if [[ ${model_flag} =~ 'pr' ]] || [[ ${model_flag} =~ 'single' ]]; then #model_flag
   echo "######  model_flag pr"
   export CUDA_VISIBLE_DEVICES=${cudaid2} #cudaid2

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
   esac

   apt-get update
   apt-get install ffmpeg -y
   apt-get install cmake -y
   apt-get install gcc -y

   unset http_proxy
   unset https_proxy
   echo "######  ----install  paddle-----"
   python -m pip install --ignore-installed  --upgrade pip \
      -i https://mirror.baidu.com/pypi/simple
   python -m pip uninstall paddlepaddle-gpu -y
   python -m pip install ${paddle_compile} #paddle_compile

else
   # ppgan
   echo "######  ffmpeg"
   yum update -y
   yum install epel-release -y
   rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
   rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
   yum install boost -y
   yum install opencv -y
   yum install ffmpeg -y
   ffmpeg
   #install  dlib
   echo "######  gcc"
   yum install gcc -y
   yum install centos-release-scl -y
   yum install devtoolset-8-gcc -y
   ls /opt/rh/devtoolset-8/enable
   source /opt/rh/devtoolset-8/enable
   gcc -v
   echo "######  cmake"
   yum install cmake -y
   cmake -version
fi

# paddle
echo "######  paddle version"
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

# python
python -c 'import sys; print(sys.version_info[:])'
echo "######  python version"

# env
# dependency
if [ -d "/etc/redhat-release" ]; then
   echo "######  system centos"
else
   echo "######  system linux"
fi

unset http_proxy
unset https_proxy
export FLAGS_fraction_of_gpu_memory_to_use=0.8
python -m pip install --ignore-installed  --upgrade pip \
   -i https://mirror.baidu.com/pypi/simple
echo "######  install ppgan "
python -m pip install  ppgan \
   -i https://mirror.baidu.com/pypi/simple
python -m pip install  -v -e. -i https://mirror.baidu.com/pypi/simple
echo "######  install dlib "
# python -m pip install --ignore-installed  dlib
python -m pip install  dlib \
   -i https://mirror.baidu.com/pypi/simple
# python -m pip install data/dlib-19.22.1-cp37-cp37m-linux_x86_64.whl
# python -m pip install data/dlib-19.22.99-cp38-cp38-linux_x86_64.whl
python -c 'import dlib'
python -m pip install -r requirements.txt  \
   -i https://mirror.baidu.com/pypi/simple

echo "######  install done "


pip list
echo "######  pip list"

# dir
log_path=log
stage_list='train eval infer'
for stage in  ${stage_list}
do
if [ -d ${log_path}/${stage} ]; then
   echo -e "\033[33m ${log_path}/${stage} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${stage}
   echo -e "\033[33m ${log_path}/${stage} is created successfully!\033[0m"
fi
done

rm -rf models_list
rm -rf models_list_all

find configs/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'\
   | grep -v 'wav2lip' | grep -v 'edvr_l_blur_wo_tsa' | grep -v 'edvr_l_blur_w_tsa' | grep -v 'mprnet_deblurring' | grep -v 'msvsr_l_reds' \
   > models_list_all

if [[ ${model_flag} =~ 'CI_all' ]]; then
   shuf models_list_all > models_list
elif [[ ${model_flag} =~ "pr" ]];then
   shuf -n ${pr_num} models_list_all > models_list
elif [[ ${model_flag} =~ "single" ]];then
   echo $1 > models_list
else
   shuf models_list_all > models_list
fi

echo "######  length models_list"
wc -l models_list
cat models_list
if [[ ${model_flag} =~ "pr" ]];then
   # git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
      # | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list
   git diff $(git log --pretty=oneline |grep "#"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
      | grep diff|grep yaml|grep configs|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
   echo "######  diff models_list_diff"
   wc -l models_list_diff
   cat models_list_diff
   shuf -n 5 models_list_diff >> models_list #防止diff yaml文件过多导致pr时间过长
fi
cat models_list | sort | uniq > models_list_run_tmp  #去重复
shuf models_list_run_tmp > models_list_run
rm -rf models_list_run_tmp
echo "######  run models_list"
wc -l models_list_run
cat models_list_run

cat models_list_run | while read line
do
echo $line
filename=${line##*/}
model=${filename%.*}
if [ -d "output" ]; then
   rm -rf output
fi
sed -i '1s/epochs/total_iters/' $line
# animeganv2
sed -i 's/pretrain_ckpt:/pretrain_ckpt: #/g' $line
case ${model} in
lapstyle_draft|lapstyle_rev_first|lapstyle_rev_second|singan_finetune|singan_animation|singan_sr|singan_universal)
python tools/main.py --config-file $line \
   -o total_iters=20 snapshot_config.interval=10 log_config.interval=1 output_dir=output \
   > $log_path/train/${model}.log 2>&1
params_dir=$(ls output)
echo "######  params_dir"
echo $params_dir
if [ -f "output/$params_dir/iter_20_checkpoint.pdparams" ];then
   echo -e "\033[33m train of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/train/${model}.log
   echo -e "\033[31m train of $model failed!\033[0m"| tee -a $log_path/result.log
fi
  ;;
*)

if [[ ! ${line} =~ 'makeup' ]]; then
   python  -m paddle.distributed.launch tools/main.py --config-file $line \
      -o total_iters=20 snapshot_config.interval=10 log_config.interval=1 output_dir=output dataset.train.batch_size=1 \
      > $log_path/train/${model}.log 2>&1
else
   python  -m paddle.distributed.launch tools/main.py --config-file $line \
      -o total_iters=20 snapshot_config.interval=10 log_config.interval=1 output_dir=output \
      > $log_path/train/${model}.log 2>&1
fi
params_dir=$(ls output)
echo "######  params_dir"
echo $params_dir
if [ -f "output/$params_dir/iter_20_checkpoint.pdparams" ];then
   echo -e "\033[33m train of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/train/${model}.log
   echo -e "\033[31m train of $model failed!\033[0m"| tee -a $log_path/result.log
fi
  ;;
esac

# evaluate
ls output/$params_dir/
echo "model is ${model}"
case ${model} in
stylegan_v2_256_ffhq)
  python tools/extract_weight.py output/$params_dir/iter_20_checkpoint.pdparams \
   --net-name gen_ema \
   --output stylegan_extract.pdparams \
   > $log_path/eval/${model}_extract_weight.log 2>&1
  if [[ $? -eq 0 ]];then
     echo -e "\033[33m extract_weight of $model  successfully!\033[0m"| tee -a $log_path/result.log
  else
     cat $log_path/eval/${model}.log
     echo -e "\033[31m extract_weight of $model failed!\033[0m"| tee -a $log_path/result.log
  fi
  python applications/tools/styleganv2.py --output_path stylegan_infer \
   --weight_path stylegan_extract.pdparams \
   --size 256 \
   > $log_path/eval/${model}.log 2>&1
  if [[ $? -eq 0 ]];then
     echo -e "\033[33m evaluate of $model  successfully!\033[0m"| tee -a $log_path/result.log
  else
     cat $log_path/eval/${model}.log
     echo -e "\033[31m evaluate of $model failed!\033[0m"| tee -a $log_path/result.log
  fi
  ;;
makeup)
   echo "skip eval makeup"
  sleep 0.01
  ;;
msvsr_l_reds)
   echo "skip eval msvsr_l_reds"
  sleep 0.01
  ;;
*)
# echo $params_dir
  python tools/main.py --config-file $line \
   --evaluate-only --load output/$params_dir/iter_20_checkpoint.pdparams \
   > $log_path/eval/${model}.log 2>&1
  if [[ $? -eq 0 ]];then
     echo -e "\033[33m evaluate of $model  successfully!\033[0m"| tee -a $log_path/result.log
  else
     cat $log_path/eval/${model}.log
     echo -e "\033[31m evaluate of $model failed!\033[0m"| tee -a $log_path/result.log
  fi
  ;;
esac
done

if [[ ! ${model_flag} =~ "single" ]];then
   #infer
   python -u applications/tools/styleganv2.py \
      --output_path styleganv2_infer \
      --model_type ffhq-config-f \
      --seed 233 \
      --size 1024 \
      --style_dim 512 \
      --n_mlp 8 \
      --channel_multiplier 2 \
      --n_row 3 \
      --n_col 5 \
      > $log_path/infer/styleganv2.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of styleganv2  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/styleganv2.log
      echo -e "\033[31m infer of styleganv2 failed!\033[0m"| tee -a $log_path/result.log
   fi
   # Wav2Lip
   # python applications/tools/wav2lip.py --face ./docs/imgs/mona7s.mp4 --audio ./docs/imgs/guangquan.m4a --outfile Wav2Lip_infer.mp4 > $log_path/infer/wav2lip.log 2>&1
   # if [[ $? -eq 0 ]];then
   #    echo -e "\033[33m infer of wav2lip  successfully!\033[0m"| tee -a $log_path/result.log
   # else
   #    cat $log_path/infer/wav2lip.log
   #    echo -e "\033[31m infer of wav2lip failed!\033[0m"| tee -a $log_path/result.log
   # fi
   # animeganv2
   python applications/tools/animeganv2.py \
      --input_image ./docs/imgs/animeganv2_test.jpg \
      > $log_path/infer/animeganv2.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of animeganv2  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/animeganv2.log
      echo -e "\033[31m infer of animeganv2 failed!\033[0m"| tee -a $log_path/result.log
   fi
   # fist order motion model
   python -u applications/tools/first-order-demo.py \
      --driving_video ./docs/imgs/fom_dv.mp4 \
      --source_image ./docs/imgs/fom_source_image.png \
      --ratio 0.4 \
      --relative \
      --adapt_scale \
      > $log_path/infer/fist_order_motion_model.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of fist order motion model  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/fist_order_motion_model.log
      echo -e "\033[31m infer of fist order motion model failed!\033[0m"| tee -a $log_path/result.log
   fi

   if [[ ! ${model_flag} == "pr" ]];then
      # fist order motion model multi_person
      python -u applications/tools/first-order-demo.py \
         --driving_video ./docs/imgs/fom_dv.mp4 \
         --source_image ./docs/imgs/fom_source_image_multi_person.jpg \
         --ratio 0.4 \
         --relative \
         --adapt_scale \
         --multi_person \
         > $log_path/infer/fist_order_motion_model_multi_person.log 2>&1
      if [[ $? -eq 0 ]];then
         echo -e "\033[33m infer of fist order motion model  multi_person successfully!\033[0m"| tee -a $log_path/result.log
      else
         cat $log_path/infer/fist_order_motion_model.log
         echo -e "\033[31m infer of fist order motion model multi_person failed!\033[0m"| tee -a $log_path/result.log
      fi
   fi

   # face_parse
   python applications/tools/face_parse.py --input_image ./docs/imgs/face.png > $log_path/infer/face_parse.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of face_parse  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/face_parse.log
      echo -e "\033[31m infer of face_parse failed!\033[0m"| tee -a $log_path/result.log
   fi
   # psgan
   python tools/psgan_infer.py \
      --config-file configs/makeup.yaml \
      --source_path  docs/imgs/ps_source.png \
      --reference_dir docs/imgs/ref \
      --evaluate-only \
      > $log_path/infer/psgan.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of psgan  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/psgan.log
      echo -e "\033[31m infer of psgan failed!\033[0m"| tee -a $log_path/result.log
   fi

   # video restore
   python applications/tools/video-enhance.py \
      --input data/Peking_input360p_clip_10_11.mp4 \
      --process_order DAIN DeOldify EDVR \
      --output video_restore_infer \
      > $log_path/infer/video_restore.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of video restore  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/video_restore.log
      echo -e "\033[31m infer of video restore failed!\033[0m"| tee -a $log_path/result.log
   fi
fi

# result
num=`cat $log_path/result.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
echo -e "-----------------------------base cases-----------------------------"
cat $log_path/result.log | grep "failed"
echo -e "--------------------------------------------------------------------"
exit 1
else
exit 0
fi
