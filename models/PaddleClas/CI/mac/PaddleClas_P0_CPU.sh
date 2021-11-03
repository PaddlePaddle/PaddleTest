unset GREP_OPTIONS

# data
rm -rf dataset
ln -s ${data_path} dataset


unset http_proxy
unset https_proxy
# env
python -m pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple  
python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
python -m pip install paddleslim -i https://mirror.baidu.com/pypi/simple  

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
python tools/train.py  -c $line -o Global.epochs=2 -o DataLoader.Train.sampler.batch_size=32 -o DataLoader.Eval.sampler.batch_size=32 -o Global.device=cpu > $log_path/train/$model.log 2>&1 
if [ $? -eq 0 ];then
   echo -e "\033[33m training of $model  successfully!\033[0m"|tee -a $log_path/result.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m"|tee -a $log_path/result.log
fi 
sleep 5

# eval
python tools/eval.py -c $line -o Global.pretrained_model=$output_path/$model/latest -o Global.device=cpu > $log_path/eval/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m eval of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/eval/$model.log
   echo -e "\033[31m eval of $model failed!\033[0m" | tee -a $log_path/result.log
fi

# infer
python tools/infer.py -c $line -o Global.pretrained_model=$output_path/$model/latest -o Global.device=cpu > $log_path/infer/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/${model}_infer.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
fi

# export_model
python tools/export_model.py -c $line -o Global.pretrained_model=$output_path/$model/latest -o Global.save_inference_dir=./inference/$model -o Global.device=cpu > $log_path/export_model/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m export_model of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/export_model/$model.log
   echo -e "\033[31m export_model of $model failed!\033[0m" | tee -a $log_path/result.log
fi

# predict
cd deploy
if [[ ${model} =~ '384' ]];then
   sed -i 's/size: 224/size: 384/g' configs/inference_cls.yaml
   sed -i 's/resize_short: 256/resize_short: 384/g' configs/inference_cls.yaml
   python python/predict_cls.py -c configs/inference_cls.yaml  -o Global.enable_mkldnn=False -o Global.inference_model_dir="../inference/"$model -o Global.use_gpu=False > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/predict/${model}.log
      echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
   fi
   sed -i 's/size: 384/size: 224/g' configs/inference_cls.yaml
   sed -i 's/resize_short: 384/resize_short: 256/g' configs/inference_cls.yaml
else
   python python/predict_cls.py -c configs/inference_cls.yaml  -o Global.enable_mkldnn=False -o Global.inference_model_dir="../inference/"$model -o Global.use_gpu=False > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/${model}.log
      echo -e "\033[31m predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
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
