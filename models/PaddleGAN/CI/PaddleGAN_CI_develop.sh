unset GREP_OPTIONS
echo ${cudaid1}
echo ${cudaid2}
echo ${Data_path}
echo ${paddle_compile}
export CUDA_VISIBLE_DEVICES=${cudaid2}

# data
echo "----ln  data-----"
rm -rf data
ln -s ${Data_path} data
ls data

if [ ! -f "/root/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz" ]; then
   wget -P /root/.cache/paddle/dataset/mnist/ https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz
fi

if [ ! -f "/root/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz" ]; then
   wget -P /root/.cache/paddle/dataset/mnist/ https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz
fi

if [[ $1 =~ 'pr' ]]; then #model_flag
   export CUDA_VISIBLE_DEVICES=$4 #cudaid
   
   echo "---py37  env -----"
   rm -rf /usr/local/python2.7.15/bin/python
   rm -rf /usr/local/bin/python
   export PATH=/usr/local/bin/python:${PATH}
   case $3 in #python
   36)
   ln -s /usr/local/bin/python3.6 /usr/local/bin/python
   ;;
   37)
   ln -s /usr/local/bin/python3.7 /usr/local/bin/python
   ;;
   38)
   ln -s /usr/local/bin/python3.8 /usr/local/bin/python
   ;;
   39)
   ln -s /usr/local/bin/python3.9 /usr/local/bin/python
   ;;
   esac
   python -c "import sys; print('python version:',sys.version_info[:])";
   
   echo "----install  paddle-----"
   python -m pip uninstall paddlepaddle-gpu -y
   python -m pip install $5 #paddle_compile

   echo "----ln  data-----"
   rm -rf data
   ln -s $6 data #data_path
   ls data
else
   # ppgan
   yum update -y
   yum install epel-release -y
   rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
   rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
   yum install cmake boost -y
   yum install opencv opencv-python opencv-devel python-devel numpy -y
   yum install ffmpeg -y
   echo "ffmpeg"
   ffmpeg
   #install  dlib
   yum install gcc gcc-c++
   yum install cmake boost
fi

# python
python -c 'import sys; print(sys.version_info[:])'
echo "python version"

# env
# dependency
if [ -d "/etc/redhat-release" ]; then
   echo "system centos"
else
   echo "system linux"
fi
export FLAGS_fraction_of_gpu_memory_to_use=0.8
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --ignore-installed  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install ppgan
python -m pip install -v -e.
python -m pip install dlib -i https://pypi.tuna.tsinghua.edu.cn/simple
# python -m pip install data/dlib-19.22.1-cp37-cp37m-linux_x86_64.whl
# python -m pip install data/dlib-19.22.99-cp38-cp38-linux_x86_64.whl

pip list
echo "pip list"

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

if [ -f "models_list" ]; then
   rm -rf models_list
   rm -rf models_list_all
fi

find configs/ -name *.yaml -exec ls -l {} \; | awk '{print $NF;}'| grep -v 'wav2lip' | grep -v 'edvr_l_blur_wo_tsa' | grep -v 'edvr_l_blur_w_tsa' | grep -v 'mprnet_deblurring' > models_list_all

if [[ ${model_flag} =~ 'CI_all' ]]; then
   shuf models_list_all > models_list
elif [[ $1 =~ "pr" ]];then
   shuf -n $2 models_list_all > models_list
else
   shuf models_list_all > models_list
fi

echo "length models_list"
wc -l models_list
git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list
echo "diff models_list"
wc -l models_list
cat models_list

cat models_list | while read line
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
lapstyle_draft|lapstyle_rev_first|lapstyle_rev_second)
python tools/main.py --config-file $line -o total_iters=20 snapshot_config.interval=10 log_config.interval=1 output_dir=output > $log_path/train/${model}.log 2>&1
params_dir=$(ls output)
echo "params_dir"
echo $params_dir
if [ -f "output/$params_dir/iter_20_checkpoint.pdparams" ];then
   echo -e "\033[33m train of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/train/${model}.log
   echo -e "\033[31m train of $model failed!\033[0m"| tee -a $log_path/result.log
fi
  ;;
*)
python  -m paddle.distributed.launch tools/main.py --config-file $line -o total_iters=20 snapshot_config.interval=10 log_config.interval=1 output_dir=output > $log_path/train/${model}.log 2>&1
params_dir=$(ls output)
echo "params_dir"
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
case ${model} in
stylegan_v2_256_ffhq)
  python tools/extract_weight.py output/$params_dir/iter_20_checkpoint.pdparams --net-name gen_ema --output stylegan_extract.pdparams > $log_path/eval/${model}_extract_weight.log 2>&1
  if [[ $? -eq 0 ]];then
     echo -e "\033[33m extract_weight of $model  successfully!\033[0m"| tee -a $log_path/result.log
  else
     cat $log_path/eval/${model}.log
     echo -e "\033[31m extract_weight of $model failed!\033[0m"| tee -a $log_path/result.log
  fi
  python applications/tools/styleganv2.py --output_path stylegan_infer --weight_path stylegan_extract.pdparams --size 256 > $log_path/eval/${model}.log 2>&1
  if [[ $? -eq 0 ]];then
     echo -e "\033[33m evaluate of $model  successfully!\033[0m"| tee -a $log_path/result.log
  else
     cat $log_path/eval/${model}.log
     echo -e "\033[31m evaluate of $model failed!\033[0m"| tee -a $log_path/result.log
  fi
  ;;
makeup)
  sleep 0.01
  ;;
*)

# echo $params_dir
  python tools/main.py --config-file $line --evaluate-only --load output/$params_dir/iter_20_checkpoint.pdparams > $log_path/eval/${model}.log 2>&1
  if [[ $? -eq 0 ]];then
     echo -e "\033[33m evaluate of $model  successfully!\033[0m"| tee -a $log_path/result.log
  else
     cat $log_path/eval/${model}.log
     echo -e "\033[31m evaluate of $model failed!\033[0m"| tee -a $log_path/result.log
  fi
  ;;
esac
done

#infer
python -u applications/tools/styleganv2.py --output_path styleganv2_infer --model_type ffhq-config-f --seed 233 --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --n_row 3 --n_col 5 > $log_path/infer/styleganv2.log 2>&1
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
python applications/tools/animeganv2.py --input_image ./docs/imgs/animeganv2_test.jpg > $log_path/infer/animeganv2.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of animeganv2  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/animeganv2.log
   echo -e "\033[31m infer of animeganv2 failed!\033[0m"| tee -a $log_path/result.log
fi
# fist order motion model
python -u applications/tools/first-order-demo.py --driving_video ./docs/imgs/fom_dv.mp4 --source_image ./docs/imgs/fom_source_image.png --ratio 0.4 --relative --adapt_scale > $log_path/infer/fist_order_motion_model.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of fist order motion model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/fist_order_motion_model.log
   echo -e "\033[31m infer of fist order motion model failed!\033[0m"| tee -a $log_path/result.log
fi

if [[! ${model_flag} =~ "CI_all" ]];then
   # fist order motion model multi_person
   python -u applications/tools/first-order-demo.py --driving_video ./docs/imgs/fom_dv.mp4 --source_image ./docs/imgs/fom_source_image_multi_person.jpg --ratio 0.4 --relative --adapt_scale --multi_person > $log_path/infer/fist_order_motion_model_multi_person.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of fist order motion model  multi_person successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/fist_order_motion_model.log
      echo -e "\033[31m infer of fist order motion model multi_person failed!\033[0m"| tee -a $log_path/result.log
   fi
fi

if [[ $1 == "pr" ]];then
   # face_parse
   python applications/tools/face_parse.py --input_image ./docs/imgs/face.png > $log_path/infer/face_parse.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of face_parse  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/face_parse.log
      echo -e "\033[31m infer of face_parse failed!\033[0m"| tee -a $log_path/result.log
   fi
   # psgan
   python tools/psgan_infer.py --config-file configs/makeup.yaml --source_path  docs/imgs/ps_source.png --reference_dir docs/imgs/ref --evaluate-only > $log_path/infer/psgan.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m infer of psgan  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/psgan.log
      echo -e "\033[31m infer of psgan failed!\033[0m"| tee -a $log_path/result.log
   fi
fi

# video restore
python applications/tools/video-enhance.py --input data/Peking_input360p_clip_10_11.mp4 --process_order DAIN DeOldify EDVR --output video_restore_infer > $log_path/infer/video_restore.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of video restore  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/video_restore.log
   echo -e "\033[31m infer of video restore failed!\033[0m"| tee -a $log_path/result.log
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
