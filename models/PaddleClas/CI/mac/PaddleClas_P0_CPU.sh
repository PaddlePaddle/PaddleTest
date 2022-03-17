unset GREP_OPTIONS
echo "Project_path"
echo ${Project_path}
echo "data_path"
echo ${data_path}

echo "path before"
pwd
if [[ ${model_flag} =~ 'CE' ]]; then
   cd ${Project_path}
   echo "path after"
   pwd
   export FLAGS_cudnn_deterministic=True
   unset FLAGS_use_virtual_memory_auto_growth
   echo $1 > clas_models_list_P0 #传入参数
fi

# data
rm -rf dataset
ln -s ${data_path} dataset

unset http_proxy
unset https_proxy

# install paddle & get yaml list
if [[ ${model_flag} =~ "pr" ]]; then
   echo "######  model_flag pr"

   echo "######  ----install  paddle-----"
   python -m pip install --ignore-installed  --upgrade pip -i https://mirror.baidu.com/pypi/simple
   python -m pip uninstall paddlepaddle-gpu -y
   python -m pip install ${paddle_compile}

   echo ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml >clas_models_list_P0
fi

# env
export FLAGS_fraction_of_gpu_memory_to_use=0.8
python -m pip install --ignore-installed --upgrade \
   pip -i https://mirror.baidu.com/pypi/simple
python -m pip install  --ignore-installed paddleslim \
   -i https://mirror.baidu.com/pypi/simple
# python -m pip install --ignore-installed dataset/visualdl-2.2.1-py3-none-any.whl \
#    -i https://mirror.baidu.com/pypi/simple
python -m pip install  -r requirements.txt  \
   -i https://mirror.baidu.com/pypi/simple

if [ -d "log" ]; then
   rm -rf log
fi
# dir
log_path=log
phases='train eval infer export_model predict'
for phase in $phases
do
if [[ -d ${log_path}/${phase} ]]; then
   echo -e "\033[33m ${log_path}/${phase} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${phase}
   echo -e "\033[33m ${log_path}/${phase} is created successfully!\033[0m"
fi
done

output_path="output"
cat clas_models_list_P0 | while read line
do
#echo $line
filename=${line##*/}
#echo $filename
model=${filename%.*}
echo $model

if [[ ${model_flag} =~ "CE" ]]; then
   if [[ ${line} =~ 'GoogLeNet' ]] || [[ ${line} =~ 'VGG' ]] || [[ ${line} =~ 'ViT' ]] || [[ ${line} =~ 'PPLCNet' ]] || [[ ${line} =~ 'MobileNetV3' ]]; then
   sed -i '' 's/learning_rate:/learning_rate: 0.0001 #/g' $line #将 学习率调低为0.0001
   echo "change lr"
   fi
   sed -i '' 's/RandCropImage/ResizeImage/g' $line
   sed -ie '/RandFlipImage/d' $line
   sed -ie '/flip_code/d' $line
      # -o Global.eval_during_train=False  \
   python tools/train.py -c $line  \
      -o Global.epochs=1  \
      -o Global.seed=1234 \
      -o Global.output_dir=$output_path \
      -o DataLoader.Train.loader.num_workers=0 \
      -o DataLoader.Train.sampler.shuffle=False  \
      -o Global.eval_interval=1  \
      -o Global.save_interval=1 \
      -o DataLoader.Train.sampler.batch_size=32 \
      -o DataLoader.Eval.sampler.batch_size=32 \
      -o Global.device=cpu \
      > $log_path/train/${model}_cpu.log 2>&1
else
   python tools/train.py  -c $line \
      -o Global.epochs=1 \
      -o DataLoader.Train.sampler.batch_size=32 \
      -o DataLoader.Eval.sampler.batch_size=32 \
      -o Global.device=cpu > $log_path/train/${model}_cpu.log 2>&1
fi
if [ $? -eq 0 ];then
   echo -e "\033[33m training of ${model}_cpu  successfully!\033[0m"|tee -a $log_path/result.log
      echo "training_exit_code: 0.0" >> $log_path/train/${model}_cpu.log
else
   cat $log_path/train/${model}_cpu.log
   echo -e "\033[31m training of ${model}_cpu failed!\033[0m"|tee -a $log_path/result.log
      echo "training_exit_code: 1.0" >> $log_path/train/${model}_cpu.log
fi
sleep 2

# eval
python tools/eval.py -c $line \
   -o Global.pretrained_model=$output_path/$model/latest \
   -o Global.device=cpu > $log_path/eval/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m eval of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "eval_exit_code: 0.0" >> $log_path/eval/$model.log
else
   cat $log_path/eval/$model.log
   echo -e "\033[31m eval of $model failed!\033[0m" | tee -a $log_path/result.log
   echo "eval_exit_code: 1.0" >> $log_path/eval/$model.log
fi

# infer
python tools/infer.py -c $line \
   -o Global.pretrained_model=$output_path/$model/latest \
   -o Global.device=cpu > $log_path/infer/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 0.0" >> $log_path/infer/$model.log
else
   cat $log_path/infer/${model}_infer.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 1.0" >> $log_path/infer/$model.log
fi

# export_model
python tools/export_model.py -c $line \
   -o Global.pretrained_model=$output_path/$model/latest \
   -o Global.save_inference_dir=./inference/$model \
   -o Global.device=cpu > $log_path/export_model/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m export_model of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "export_exit_code: 0.0" >> $log_path/export_model/$model.log
else
   cat $log_path/export_model/$model.log
   echo -e "\033[31m export_model of $model failed!\033[0m" | tee -a $log_path/result.log
   echo "export_exit_code: 1.0" >> $log_path/export_model/$model.log
fi

# predict
cd deploy
if [[ ${model} =~ '384' ]] && [[ ! ${model} =~ 'LeViT' ]];then
   sed -i '' 's/size: 224/size: 384/g' configs/inference_cls.yaml
   sed -i '' 's/resize_short: 256/resize_short: 384/g' configs/inference_cls.yaml
   python python/predict_cls.py -c configs/inference_cls.yaml  \
      -o Global.enable_mkldnn=False \
      -o Global.inference_model_dir="../inference/"$model \
      -o Global.use_gpu=False > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
      echo "predict_exit_code: 0.0" >> ../$log_path/predict/$model.log
   else
      cat ../$log_path/predict/${model}.log
      echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
      echo "predict_exit_code: 1.0" >> ../$log_path/predict/$model.log
   fi
   sed -i '' 's/size: 384/size: 224/g' configs/inference_cls.yaml
   sed -i '' 's/resize_short: 384/resize_short: 256/g' configs/inference_cls.yaml
else
   python python/predict_cls.py -c configs/inference_cls.yaml  \
      -o Global.enable_mkldnn=False \
      -o Global.inference_model_dir="../inference/"$model \
      -o Global.use_gpu=False > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
      echo "predict_exit_code: 0.0" >> ../$log_path/predict/$model.log
   else
      cat ../$log_path/predict/${model}.log
      echo -e "\033[31m predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
      echo "predict_exit_code: 1.0" >> ../$log_path/predict/$model.log
   fi
fi
cd ..
done

# PaddleClas_rec
# source PaddleClas_rec_CPU.sh

num=`cat $log_path/result.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
echo -e "-----------------------------base cases-----------------------------"
cat $log_path/result.log | grep "failed"
echo -e "--------------------------------------------------------------------"
exit 1
else
exit 0
fi
