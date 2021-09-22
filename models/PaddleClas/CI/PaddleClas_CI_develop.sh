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
python -m pip install -r requirements.txt --ignore-installed  -i https://pypi.tuna.tsinghua.edu.cn/simple

unset http_proxy
unset https_proxy


find ppcls/configs/ImageNet/ -name *.yaml -exec ls -l {} \; | awk '{print $NF;}'| grep -v 'eval' | grep -v 'kunlun' | grep -v 'distill'| grep -v 'ResNet50_fp16_dygraph' | grep -v 'ResNet50_fp16'  | grep -v 'SE_ResNeXt101_32x4d_fp16'  > models_list_all
shuf models_list_all > models_list
echo "length models_list"
wc -l models_list
cat models_list
git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list

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

cat models_list | while read line
do
#echo $line
filename=${line##*/}
#echo $filename
model=${filename%.*}
if [ -d "output" ]; then
   rm -rf output
fi
echo $model 
export CUDA_VISIBLE_DEVICES=${cudaid2}
#看情况加判断针对占大显存，MV3设置batch_size与epoch
case ${model} in
ViT_large_patch16_384|ResNeXt101_32x48d_wsl|ViT_huge_patch16_224|RedNet152|EfficientNetB6|EfficientNetB7)
python -m paddle.distributed.launch --gpus=${cudaid2} tools/train.py  -c $line -o Global.epochs=4 -o Global.output_dir=output -o DataLoader.Train.sampler.batch_size=1 -o DataLoader.Eval.sampler.batch_size=1  > $log_path/train/$model.log 2>&1
params_dir=$(ls output)
echo "params_dir"
echo $params_dir
if [ -f "output/$params_dir/latest.pdparams" ];then
   echo -e "\033[33m training of $model  successfully!\033[0m"|tee -a $log_path/result.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m"|tee -a $log_path/result.log
fi
  ;;
*)
python -m paddle.distributed.launch --gpus=${cudaid2} tools/train.py  -c $line -o Global.epochs=1 -o Global.output_dir=output -o DataLoader.Train.sampler.batch_size=8 -o DataLoader.Eval.sampler.batch_size=1  > $log_path/train/$model.log 2>&1
params_dir=$(ls output)
echo "params_dir"
echo $params_dir
if [ -f "output/$params_dir/latest.pdparams" ];then
   echo -e "\033[33m training of $model  successfully!\033[0m"|tee -a $log_path/result.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m"|tee -a $log_path/result.log
fi
  ;;
esac

export CUDA_VISIBLE_DEVICES=${cudaid1}
ls output/$params_dir/
# eval
python tools/eval.py -c $line -o Global.pretrained_model=output/$params_dir/latest > $log_path/eval/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m eval of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/eval/$model.log
   echo -e "\033[31m eval of $model failed!\033[0m" | tee -a $log_path/result.log
fi

# infer
python tools/infer.py -c $line -o Global.pretrained_model=output/$params_dir/latest > $log_path/infer/$model.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/${model}_infer.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
fi

# export_model
python tools/export_model.py -c $line -o Global.pretrained_model=output/$params_dir/latest -o Global.save_inference_dir=./inference/$model > $log_path/export_model/$model.log 2>&1
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

   python python/predict_cls.py -c configs/inference_cls.yaml -o Global.infer_imgs="./images"  -o Global.batch_size=4 -o Global.inference_model_dir="../inference/"$model > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m multi_batch_size predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/predict/${model}.log
      echo -e "\033[31m multi_batch_size predict of $model failed!\033[0m"| tee -a $log_path/result.log
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

   python python/predict_cls.py -c configs/inference_cls.yaml  -o Global.infer_imgs="./images"  -o Global.batch_size=4 -o Global.inference_model_dir="../inference/"$model > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m multi_batch_size predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/${model}.log
      echo -e "\033[31m multi_batch_size predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
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
