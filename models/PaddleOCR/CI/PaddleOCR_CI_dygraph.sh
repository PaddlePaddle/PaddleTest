unset GREP_OPTIONS
echo ${cudaid1}
echo ${cudaid2}
echo ${Data_path}
echo ${paddle_compile}
export CUDA_VISIBLE_DEVICES=${cudaid2}

if [[ ${model_flag} =~ 'CI' ]]; then
   # data
   echo "######  ----ln  data-----"
   rm -rf train_data
   rm -rf pretrain_models
   ln -s ${Data_path}/* .
   ls train_data
   gpu_flag=True
fi

if [[ $1 =~ 'pr' ]] || [[ $1 =~ 'all' ]] || [[ $1 =~ 'single' ]]; then #model_flag
   echo "######  model_flag pr"
   export CUDA_VISIBLE_DEVICES=$4 #cudaid

   echo "######  ---py37  env -----"
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

   unset http_proxy
   unset https_proxy
   echo "######  ----install  paddle-----"
   python -m pip uninstall paddlepaddle-gpu -y
   python -m pip install $5 #paddle_compile

   echo "######  ----ln  data-----"
   rm -rf pretrain_models
   rm -rf train_data
   ln -s $6/* .
   ls train_data
fi

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
# dependency
python -m pip install --ignore-installed --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip install --ignore-installed -r requirements.txt -i https://mirror.baidu.com/pypi/simple
# paddleocr
python -m pip install  --ignore-installed paddleocr -i https://mirror.baidu.com/pypi/simple
python -m pip install  --ignore-installed gast==0.3.3 -i https://mirror.baidu.com/pypi/simple

# dir
log_path=log
stage_list='train eval infer export predict'
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
rm -rf models_list_det_db
rm -rf models_list_rec

find configs/det -name '*.yml' -exec ls -l {} \; | awk '{print $NF;}'| grep 'db' > models_list_det_db
find configs/rec -name '*.yml' -exec ls -l {} \; | awk '{print $NF;}' | grep -v 'rec_multi_language_lite_train' | grep -v 'rec_r31_sar' | grep -v 'rec_resnet_stn_bilstm_att' > models_list_rec

shuf models_list_det_db > models_list_all
shuf models_list_rec >> models_list_all

if [[ ${model_flag} =~ 'CI_all' ]]; then
   shuf models_list_all > models_list
elif [[ $1 =~ "pr" ]];then
   shuf -n $2 models_list_all > models_list
elif [[ ${1} =~ "single" ]];then
   echo $7 > models_list
else
   shuf models_list_all > models_list
fi

echo "######  length models_list"
wc -l models_list
cat models_list
if [[ ${1} =~ "pr" ]];then
   # git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR | grep diff|grep yml|awk -F 'b/' '{print$2}'|tee -a  models_list
   git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR | grep diff|grep yml|grep configs|grep rec|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
   git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR | grep diff|grep yml|grep configs|grep det|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
   echo "######  diff models_list_diff"
   wc -l models_list_diff
   cat models_list_diff
   shuf -n 5 models_list_diff >> models_list #防止diff yaml文件过多导致pr时间过长
fi
echo "######  diff models_list"
wc -l models_list
cat models_list

cat models_list | while read line
do
echo $line
sed -i 's!data_lmdb_release/training!data_lmdb_release/validation!g' $line
algorithm=$(grep -i algorithm $line |awk -F: '{print $2}'| sed 's/ //g')
algorithm=`echo $algorithm | awk '{print $1}'`
echo $algorithm
filename=${line##*/}
model=${filename%.*}

if [[ $model =~ "rec" ]];then
   category="rec"
else
   category="det"
fi
echo "######  category"
echo $category

if [ ${category} == "rec" ];then

python -m paddle.distributed.launch  tools/train.py -c $line -o Train.loader.batch_size_per_card=2 Global.use_gpu=${gpu_flag} Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=200 Global.print_batch_step=10 Global.save_model_dir="output/"${model} > $log_path/train/$model.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/train/$model.log) -eq 0 ]]  && [ -f "output/"${model}"/latest.pdparams" ];then
   echo -e "\033[33m training of $model  successfully!\033[0m" | tee -a $log_path/result.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m" | tee -a $log_path/result.log
fi

else
python -m paddle.distributed.launch  tools/train.py -c $line  -o Train.loader.batch_size_per_card=2 Global.use_gpu=${gpu_flag} Global.epoch_num=1 Global.save_epoch_step=1 Global.save_model_dir="output/"${model}  > $log_path/train/$model.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/train/$model.log) -eq 0 ]]  && [ -f "output/"${model}"/latest.pdparams" ];then
   echo -e "\033[33m training of $model  successfully!\033[0m" | tee -a $log_path/result.log
else
   cat  $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m" | tee -a $log_path/result.log
fi
fi

# eval
if [[ ${model} =~ "sast" ]];then
   sleep 0.01
else
python tools/eval.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest.pdparams" > $log_path/eval/$model.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/eval/$model.log) -eq 0 ]];then
   echo -e "\033[33m eval of $model  successfully!\033[0m" | tee -a $log_path/result.log
else
   cat $log_path/eval/$model.log
   echo -e "\033[31m eval of $model failed!\033[0m" | tee -a $log_path/result.log
fi
fi

# infer
if [ ${category} == "rec" ];then
python tools/infer_${category}.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest" Global.infer_img=doc/imgs_words/en/word_1.png > $log_path/infer/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/infer/${model}.log) -eq 0 ]];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/${model}.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
fi
else
python tools/infer_${category}.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest" Global.infer_img="./doc/imgs_en/" Global.test_batch_size_per_card=1 > $log_path/infer/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/infer/${model}.log) -eq 0 ]];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/${model}.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
fi
fi

# export_model
python tools/export_model.py -c $line -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest"  Global.save_inference_dir="./models_inference/"${model} >  $log_path/export/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/export/${model}.log) -eq 0 ]];then
   echo -e "\033[33m export_model of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/export/${model}.log
   echo -e "\033[31m export_model of $model failed!\033[0m"| tee -a $log_path/result.log
fi

# predict
if [ ${category} == "rec" ];then
echo "######  rec"
if [[ $(echo $model | grep -c "chinese") -eq 0 ]];then
echo "######  none chinese"
if [ ${algorithm} == "SRN" ];then
echo "######  SRN"
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="1, 64,256" --rec_char_type="en" --rec_algorithm=${algorithm}  > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

elif [ ${algorithm} == "SAR" ];then
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 48, 48, 160" --rec_char_type="ch" --rec_char_dict_path=ppocr/utils/en_dict.txt --rec_algorithm=${algorithm}  > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

else
echo "######  none SRN"
if [[ $(echo $model | grep -c "lite") -eq 0 ]];then

if [ -f "./models_inference/${model}/inference.pdiparams"  ];then
# python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_type="en" --rec_algorithm=${algorithm}  > $log_path/predict/${model}.log 2>&1
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_char_dict_path=ppocr/utils/en_dict.txt --rec_algorithm=${algorithm}  > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

else
echo "######  have Teacher Student "
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model}"/Student" --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_char_dict_path=ppocr/utils/en_dict.txt --rec_algorithm=${algorithm} > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m Student predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m Student predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model}"/Teacher" --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_char_dict_path=ppocr/utils/en_dict.txt --rec_algorithm=${algorithm} > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m Teacher predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m Teacher predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

fi

else
echo "######  multi_language"
# multi_language
language=`echo $model |awk -F '_' '{print $2}'`
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_type="en" --rec_algorithm=${algorithm}  --rec_char_type=$language --rec_char_dict_path="ppocr/utils/dict/"$language"_dict.txt" > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

fi
fi

else
echo "######  chinese"

if [ -f "./models_inference/${model}/inference.pdiparams"  ];then
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_algorithm=${algorithm} > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   cat $log_path/predict/${model}.log
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

else
echo "######  have Teacher Student "
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model}"/Teacher" --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_algorithm=${algorithm} > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   cat $log_path/predict/${model}.log
   echo -e "\033[33m Teacher predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m Teacher predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model}"/Student" --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_algorithm=${algorithm} > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   cat $log_path/predict/${model}.log
   echo -e "\033[33m Student predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m Student predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

fi
fi

else
echo "######  det"
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./models_inference/"${model} --det_algorithm=${algorithm} > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   cat $log_path/predict/${model}.log
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

fi
done

# cls
if [[ ${model_flag} =~ 'CI_all' ]]; then
   python -m paddle.distributed.launch  tools/train.py -c configs/cls/cls_mv3.yml -o Train.loader.batch_size_per_card=2 Global.print_batch_step=1 Global.epoch_num=1 > $log_path/train/cls_mv3.log 2>&1
   if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/train/cls_mv3.log) -eq 0 ]];then
      echo -e "\033[33m training of cls_mv3  successfully!\033[0m" | tee -a $log_path/result.log
   else
      cat  $log_path/train/cls_mv3.log
      echo -e "\033[31m training of cls_mv3 failed!\033[0m" | tee -a $log_path/result.log
   fi
   python tools/eval.py -c configs/cls/cls_mv3.yml -o Global.checkpoints=output/cls/mv3/latest > $log_path/eval/cls_mv3.log 2>&1
   if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/eval/cls_mv3.log) -eq 0 ]];then
      echo -e "\033[33m eval of cls_mv3  successfully!\033[0m" | tee -a $log_path/result.log
   else
      cat $log_path/eval/cls_mv3.log
      echo -e "\033[31m eval of cls_mv3 failed!\033[0m" | tee -a $log_path/result.log
   fi
   python tools/infer_cls.py -c configs/cls/cls_mv3.yml -o Global.pretrained_model=output/cls/mv3/latest Global.load_static_weights=false Global.infer_img=doc/imgs_words/ch/word_1.jpg > $log_path/infer/cls_mv3.log 2>&1
   if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/infer/cls_mv3.log) -eq 0 ]];then
      echo -e "\033[33m infer of cls_mv3  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/infer/cls_mv3.log
      echo -e "\033[31m infer of cls_mv3 failed!\033[0m"| tee -a $log_path/result.log
   fi
fi

num=`cat $log_path/result.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
echo -e "-----------------------------base cases-----------------------------"
cat $log_path/result.log | grep "failed"
echo -e "--------------------------------------------------------------------"
exit 1
else
exit 0
fi
