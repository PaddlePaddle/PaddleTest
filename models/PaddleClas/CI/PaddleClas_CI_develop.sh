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
rm -rf dataset
ln -s ${Data_path} dataset

# env
export CUDA_VISIBLE_DEVICES=${cudaid2}
export FLAGS_fraction_of_gpu_memory_to_use=0.8
python -m pip install -r requirements.txt --ignore-installed

find ppcls/configs/ImageNet/ -name *.yaml -exec ls -l {} \; | awk '{print $NF;}'| grep -v 'eval'| grep -v 'ResNeXt101_32x8d_wsl'| grep -v 'kunlun' | grep -v 'distill'| grep -v 'ResNeXt101_32x16d_wsl' > models_list_all
shuf models_list_all > models_list
echo "length models_list"
wc -l models_list
cat models_list
git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print}') HEAD --diff-filter=AMR | grep diff|grep yaml|awk -F 'b/' '{print }'|tee -a  models_list

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
cat models_list | while read line
do
#echo $line
filename=${line##*/}
#echo $filename
model=${filename%.*}
rm -rf $output_path
echo $model
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch --gpus=${cudaid2} tools/train.py  -c $line -o Global.epochs=2 -o Global.output_dir=${output_path} -o DataLoader.Train.sampler.batch_size=1 -o DataLoader.Eval.sampler.batch_size=1  > $log_path/train/$model.log 2>&1 
if [ $? -eq 0 ];then
   echo -e "\033[33m training of $model  successfully!\033[0m"|tee -a $log_path/result.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m"|tee -a $log_path/result.log
fi 
sleep 5s

params_dir=$(ls $output_path)
echo $params_dir
ls $output_path/$params_dir/
export CUDA_VISIBLE_DEVICES=${cudaid1}
# eval
python tools/eval.py -c $line -o Global.pretrained_model=$output_path/$params_dir/best_model > $log_path/eval/$model.log 2>&1
if [ $? -eq 0 ];then   
   echo -e "\033[33m eval of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/eval/$model.log
   echo -e "\033[31m eval of $model failed!\033[0m" | tee -a $log_path/result.log
fi

# infer
python tools/infer.py -c $line -o Global.pretrained_model=$output_path/$params_dir/best_model > $log_path/infer/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/${model}_infer.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
fi

# export_model
python tools/export_model.py -c $line -o Global.pretrained_model=$output_path/$params_dir/best_model -o Global.save_inference_dir=./inference/$model > $log_path/export_model/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m export_model of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/export_model/$model.log
   echo -e "\033[31m export_model of $model failed!\033[0m" | tee -a $log_path/result.log
fi

cd deploy
if [[ ${model} =~ '384' ]];then
   sed -i 's/size: 224/size: 384/g' configs/inference_cls.yaml
   sed -i 's/resize_short: 256/resize_short: 384/g' configs/inference_cls.yaml
   python python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir="../inference/"$model > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/predict/${model}.log
      echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
   fi
   sed -i 's/size: 384/size: 224/g' configs/inference_cls.yaml
   sed -i 's/resize_short: 384/resize_short: 256/g' configs/inference_cls.yaml
else
   python python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir="../inference/"$model > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/${model}.log
      echo -e "\033[31m predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
   fi
fi
cd ..
done

num=`cat $log_path/result.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
echo -e "-----------------------------base cases-----------------------------"
cat $log_path/result.log | grep "failed"
echo -e "--------------------------------------------------------------------"
exit 1
else
exit 0
fi
