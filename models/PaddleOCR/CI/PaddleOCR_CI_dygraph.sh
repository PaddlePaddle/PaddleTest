unset GREP_OPTIONS
echo ${cudaid1}
echo ${cudaid2}
echo ${Data_path}
echo ${paddle_compile}

# python
python -c 'import sys; print(sys.version_info[:])'
echo "python version"

export http_proxy=${http_proxy};
export https_proxy=${https_proxy};

# data
rm -rf train_data
ln -s ${Data_path}/train_data train_data
gpu_flag=True

# env
export CUDA_VISIBLE_DEVICES=${cudaid2}
export FLAGS_fraction_of_gpu_memory_to_use=0.8

# dependency
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --ignore-installed
# paddleocr
python -m pip install paddleocr
python -m pip install gast==0.3.3

# download model
if [ ! -f "pretrain_models/MobileNetV3_large_x0_5_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ResNet18_vd_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_vd_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ResNet50_vd_ssld_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ResNet50_vd_pretrained.pdparams" ]; then
  wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
fi

if [ ! -f "pretrain_models/ch_ppocr_mobile_v2.0_det_train.tar" ]; then
    wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
    tar xf pretrain_models/ch_ppocr_mobile_v2.0_det_train.tar -C pretrain_models
fi

if [ ! -f "pretrain_models/ch_ppocr_server_v2.0_det_train.tar" ]; then
    wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
    tar xf pretrain_models/ch_ppocr_server_v2.0_det_train.tar -C pretrain_models
fi

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

find configs/det -name *.yml -exec ls -l {} \; | awk '{print $NF;}'| grep 'db' > models_list_det_db
find configs/rec -name *.yml -exec ls -l {} \; | awk '{print $NF;}' | grep -v 'rec_multi_language_lite_train' | grep -v 'rec_chinese_lite_train_distillation_v2.1' | grep -v 'rec_mtb_nrtr' > models_list_rec

shuf models_list_det_db > models_list
shuf models_list_rec >> models_list
wc -l models_list
cat models_list

cat models_list | while read line
do
echo $line
sed -i 's!data_lmdb_release/training!data_lmdb_release/validation!g' $line
algorithm=$(grep -i algorithm $line |awk -F: '{print $2}'| sed 's/ //g')
filename=${line##*/}
model=${filename%.*}
category=${model%%_*}

if [ ${category} == "ch" ];then
category="det"
fi

if [ ${category} == "rec" ];then

if [[ $(echo $model | grep -c "chinese") -ne 0 ]];then
python -m paddle.distributed.launch   tools/train.py -c $line -o Train.loader.batch_size_per_card=2 Global.distort=True Global.use_space_char=True Global.use_gpu=${gpu_flag} Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=200 Global.print_batch_step=10 Global.save_model_dir="output/"${model} > $log_path/train/$model_use_space.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/train/$model_use_space.log) -eq 0 ]];then
   echo -e "\033[33m training of $model use_space_char successfully!\033[0m" | tee -a $log_path/result.log
else
   $log_path/train/$model_use_space.log
   echo -e "\033[31m training of $model use_space_char failed!\033[0m" | tee -a $log_path/result.log
fi
fi

python -m paddle.distributed.launch  tools/train.py -c $line -o Train.loader.batch_size_per_card=2 Global.use_gpu=${gpu_flag} Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=200 Global.print_batch_step=10 Global.save_model_dir="output/"${model} > $log_path/train/$model.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/train/$model.log) -eq 0 ]];then
   echo -e "\033[33m training of $model  successfully!\033[0m" | tee -a $log_path/result.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m" | tee -a $log_path/result.log
fi

else
python -m paddle.distributed.launch  tools/train.py -c $line  -o Train.loader.batch_size_per_card=2 Global.use_gpu=${gpu_flag} Global.epoch_num=1 Global.save_epoch_step=1 Global.save_model_dir="output/"${model}  > $log_path/train/$model.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/train/$model.log) -eq 0 ]];then
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
python tools/eval.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest" > $log_path/eval/$model.log 2>&1
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
if [[ $(echo $model | grep -c "chinese") -eq 0 ]];then
if [ ${algorithm} == "SRN" ];then
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="1, 64,256" --rec_char_type="en" --rec_algorithm=${algorithm}  > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

else
if [[ $(echo $model | grep -c "lite") -eq 0 ]];then
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_type="en" --rec_algorithm=${algorithm}  > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi

else

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
python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_type="ch" --rec_algorithm=${algorithm} > $log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   cat $log_path/predict/${model}.log
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
fi
fi

else
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

# # predict
# python tools/infer/predict_system.py --image_dir="./doc/imgs/12.jpg" --det_model_dir="./models/inference/det/"  --rec_model_dir="./models/inference/rec/" > $log_path/predict/predict_system.log 2>&1
# if [[ $? -eq 0 ]]; then
#    cat $log_path/predict/predict_system.log
#    echo -e "\033[33m predict of system  successfully!\033[0m"| tee -a $log_path/result.log
# else
#    cat $log_path/predict/predict_system.log
#    echo -e "\033[31m predict of system failed!\033[0m"| tee -a $log_path/result.log
# fi

# cls
export CUDA_VISIBLE_DEVICES=${cudaid2}
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

num=`cat $log_path/result.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
echo -e "-----------------------------base cases-----------------------------"
cat $log_path/result.log | grep "failed"
echo -e "--------------------------------------------------------------------"
exit 1
else
exit 0
fi
