unset GREP_OPTIONS
echo ${cudaid1}
echo ${cudaid2}
echo ${Data_path}
echo ${paddle_compile}
export CUDA_VISIBLE_DEVICES=${cudaid2}

# data
rm -rf dataset
ln -s ${Data_path} dataset
ls dataset

cd deploy
rm -rf recognition_demo_data_v1.0
rm -rf recognition_demo_data_v1.1
rm -rf models
ln -s  ${Data_path}/* .
cd ..

if [[ $1 =~ 'pr' ]] || [[ $1 =~ 'all' ]] || [[ $1 =~ 'single' ]]; then #model_flag
   echo "model_flag pr"
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
   
   unset http_proxy
   unset https_proxy
   echo "----install  paddle-----"
   python -m pip uninstall paddlepaddle-gpu -y
   python -m pip install $5 #paddle_compile

   echo "----ln  data-----"
   rm -rf dataset
   ln -s $6 dataset #data_path
   ls dataset
   cd deploy

   rm -rf recognition_demo_data_v1.0
   rm -rf recognition_demo_data_v1.1
   rm -rf models
   ln -s $6/* .
   cd ..
fi

# python
python -c 'import sys; print(sys.version_info[:])'
echo "python version"

#system
if [ -d "/etc/redhat-release" ]; then
   echo "system centos"
else
   echo "system linux"
fi

unset http_proxy
unset https_proxy
# env
export FLAGS_fraction_of_gpu_memory_to_use=0.8 
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install  -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install  --ignore-installed paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ -f "models_list" ]; then
   rm -rf models_list
   rm -rf models_list_all
fi

find ppcls/configs/ImageNet/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'| grep -v 'eval' | grep -v 'kunlun' | grep -v 'distill'| grep -v 'ResNet50_fp16_dygraph' | grep -v 'ResNet50_fp16'  | grep -v 'SE_ResNeXt101_32x4d_fp16'  > models_list_all

if [[ ${model_flag} =~ 'CI_step1' ]]; then
   cat models_list_all | while read line
   do
   if [[ ${line} =~ 'AlexNet' ]] ||[[ ${line} =~ 'DPN' ]] ||[[ ${line} =~ 'DarkNet' ]] ||[[ ${line} =~ 'DeiT' ]] ||[[ ${line} =~ 'DenseNet' ]] ||[[ ${line} =~ 'EfficientNet' ]] ||[[ ${line} =~ 'GhostNet' ]] ||[[ ${line} =~ 'HRNet' ]] ||[[ ${line} =~ 'HarDNet' ]] ||[[ ${line} =~ 'Inception' ]] ||[[ ${line} =~ 'LeViT' ]] ||[[ ${line} =~ 'MixNet' ]] ||[[ ${line} =~ 'MobileNetV1' ]] ||[[ ${line} =~ 'MobileNetV2' ]] ||[[ ${line} =~ 'MobileNetV3' ]] ||[[ ${line} =~ 'PPLCNet' ]] ||[[ ${line} =~ 'ReXNet' ]] ||[[ ${line} =~ 'RedNet' ]] ||[[ ${line} =~ 'Res2Net' ]]; then
      echo ${line}  >> models_list
   fi
   done

elif [[ ${1} =~ "pr" ]];then
   shuf -n $2 models_list_all > models_list

elif [[ ${1} =~ "single" ]];then
   echo $7 > models_list

elif [[ ${model_flag} =~ 'CI_step2' ]]; then
   cat models_list_all | while read line
   do
   if [[ ! ${line} =~ 'AlexNet' ]] && [[ ! ${line} =~ 'DPN' ]] && [[ ! ${line} =~ 'DarkNet' ]] && [[ ! ${line} =~ 'DeiT' ]] && [[ ! ${line} =~ 'DenseNet' ]] && [[ ! ${line} =~ 'EfficientNet' ]] && [[ ! ${line} =~ 'GhostNet' ]] && [[ ! ${line} =~ 'HRNet' ]] && [[ ! ${line} =~ 'HarDNet' ]] && [[ ! ${line} =~ 'Inception' ]] && [[ ! ${line} =~ 'LeViT' ]] && [[ ! ${line} =~ 'MixNet' ]] && [[ ! ${line} =~ 'MobileNetV1' ]] && [[ ! ${line} =~ 'MobileNetV2' ]] && [[ ! ${line} =~ 'MobileNetV3' ]] && [[ ! ${line} =~ 'PPLCNet' ]] && [[ ! ${line} =~ 'ReXNet' ]] && [[ ! ${line} =~ 'RedNet' ]] && [[ ! ${line} =~ 'Res2Net' ]]; then
      echo ${line}  >> models_list
   fi
   done

else
   shuf models_list_all > models_list
   
fi

echo "length models_list"
wc -l models_list
cat models_list
if [[ ${1} =~ "pr" ]];then
   git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list
fi
echo "diff models_list"
wc -l models_list
cat models_list

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
#看情况加判断针对占大显存，MV3设置batch_size与epoch

#visualdl
echo "visualdl err"
ls /root/.visualdl/conf
rm -rf /root/.visualdl/conf

case ${model} in
ViT_large_patch16_384|ResNeXt101_32x48d_wsl|ViT_huge_patch16_224|RedNet152|EfficientNetB6|EfficientNetB7)
python -m paddle.distributed.launch --gpus=${cudaid2} tools/train.py  -c $line -o Global.epochs=2 -o Global.output_dir=output -o DataLoader.Train.sampler.batch_size=1 -o DataLoader.Eval.sampler.batch_size=1  > $log_path/train/$model.log 2>&1
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
python -m paddle.distributed.launch --gpus=${cudaid2} tools/train.py  -c $line -o Global.epochs=1 -o Global.output_dir=output -o DataLoader.Train.sampler.batch_size=4 -o DataLoader.Eval.sampler.batch_size=1  > $log_path/train/$model.log 2>&1
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

ls output/$params_dir/
# eval
python tools/eval.py -c $line -o Global.pretrained_model=output/$params_dir/latest -o DataLoader.Eval.sampler.batch_size=1 > $log_path/eval/$model.log 2>&1
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
if [[ ${model} =~ '384' ]] && [[ ! ${model} =~ 'LeViT' ]];then
   sed -i 's/size: 224/size: 384/g' configs/inference_cls.yaml
   sed -i 's/resize_short: 256/resize_short: 384/g' configs/inference_cls.yaml
   python python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir="../inference/"$model > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/${model}.log
      echo -e "\033[31m predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
   fi

   python python/predict_cls.py -c configs/inference_cls.yaml -o Global.infer_imgs="./images"  -o Global.batch_size=4 -o Global.inference_model_dir="../inference/"$model > ../$log_path/predict/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m multi_batch_size predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/${model}.log
      echo -e "\033[31m multi_batch_size predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
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

if [[ ${model_flag} =~ 'CI_step3' ]] || [[ $1 =~ 'pr' ]] ; then
   echo "rec step"

   # log
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

   # small data
   # icartoon_dataset
   sed -ie '/self.images = self.images\[:200\]/d'  ppcls/data/dataloader/icartoon_dataset.py
   sed -ie '/self.labels = self.labels\[:200\]/d'  ppcls/data/dataloader/icartoon_dataset.py
   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.images = self.images\[:200\]'  ppcls/data/dataloader/icartoon_dataset.py
   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.labels = self.labels\[:200\]'  ppcls/data/dataloader/icartoon_dataset.py

   # product_dataset 
   sed -ie '/self.images = self.images\[:200\]/d'  ppcls/data/dataloader/imagenet_dataset.py
   sed -ie '/self.labels = self.labels\[:200\]/d'  ppcls/data/dataloader/imagenet_dataset.py
   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.images = self.images\[:200\]'  ppcls/data/dataloader/imagenet_dataset.py
   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.labels = self.labels\[:200\]'  ppcls/data/dataloader/imagenet_dataset.py

   # # logo_dataset 
   # sed -ie '/self.images = self.images\[:10000\]/d'  ppcls/data/dataloader/logo_dataset.py
   # sed -ie '/self.labels = self.labels\[:10000\]/d'  ppcls/data/dataloader/logo_dataset.py
   # sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.images = self.images\[:10000\]'  ppcls/data/dataloader/logo_dataset.py
   # sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.labels = self.labels\[:200\]'  ppcls/data/dataloader/logo_dataset.py

   # vehicle_dataset
   sed -ie '/self.images = self.images\[:200\]/d'  ppcls/data/dataloader/vehicle_dataset.py
   sed -ie '/self.labels = self.labels\[:200\]/d'  ppcls/data/dataloader/vehicle_dataset.py
   sed -ie '/self.bboxes = self.bboxes\[:200\]/d'  ppcls/data/dataloader/vehicle_dataset.py
   sed -ie '/self.cameras = self.cameras\[:200\]/d'  ppcls/data/dataloader/vehicle_dataset.py

   numbers=`grep -n 'assert os.path.exists(self.images\[-1\])' ppcls/data/dataloader/vehicle_dataset.py |awk -F: '{print $1}'`
   number1=`echo $numbers |cut -d' ' -f1`
   sed -i "`echo $number1` a\        self.bboxes = self.bboxes\[:200\]" ppcls/data/dataloader/vehicle_dataset.py

   numbers=`grep -n 'assert os.path.exists(self.images\[-1\])' ppcls/data/dataloader/vehicle_dataset.py |awk -F: '{print $1}'`
   number2=`echo $numbers |cut -d' ' -f2`
   sed -i "`echo $number2` a\        self.cameras = self.cameras\[:200\]" ppcls/data/dataloader/vehicle_dataset.py

   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.images = self.images\[:200\]'  ppcls/data/dataloader/vehicle_dataset.py
   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.labels = self.labels\[:200\]'  ppcls/data/dataloader/vehicle_dataset.py


   find ppcls/configs/Cartoonface/ -name *.yaml -exec ls -l {} \; | awk '{print $NF;}' >> models_list_rec
   # find ppcls/configs/Logo/ -name *.yaml -exec ls -l {} \; | awk '{print $NF;}' >> models_list_rec
   find ppcls/configs/Products/ -name *.yaml -exec ls -l {} \; | awk '{print $NF;}' | grep  Inshop  >> models_list_rec
   find ppcls/configs/Vehicle/ -name *.yaml -exec ls -l {} \; | awk '{print $NF;}' | grep -v 'ReID' >> models_list_rec
   echo "rec models_list"
   wc -l models_list_rec
   cat models_list_rec

   if [[ $1 =~ 'pr' ]]; then
      shuf -n $2 models_list_rec > models_list
   else
      shuf models_list_rec > models_list
   fi
   echo "rec run models_list"
   cat models_list

   #train
   cat models_list | while read line
   do
   #echo $line
   filename=${line##*/}
   #echo $filename
   model=${filename%.*}
   category=`echo $line | awk -F [\/] '{print $3}'`
   echo ${category}_${model}
   echo $category
   python -m paddle.distributed.launch tools/train.py  -c $line -o Global.epochs=1 -o Global.save_interval=1 -o Global.eval_interval=1 -o Global.output_dir="./output/"${category}_${model} > $log_path/train/${category}_${model}.log 2>&1
   if [[ $? -eq 0 ]] && [[ $(grep -c -i "Error" $log_path/train/${category}_${model}.log) -eq 0 ]];then
      echo -e "\033[33m training of ${category}_${model}  successfully!\033[0m"|tee -a $log_path/result.log
   else
      cat $log_path/train/${category}_${model}.log
      echo -e "\033[31m training of ${category}_${model} failed!\033[0m"|tee -a $log_path/result.log
   fi 

   # eval
   python tools/eval.py -c $line -o Global.pretrained_model=output/${category}_${model}/RecModel/latest > $log_path/eval/${category}_${model}.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m eval of ${category}_${model}  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/eval/${category}_${model}.log
      echo -e "\033[31m eval of ${category}_${model} failed!\033[0m" | tee -a $log_path/result.log
   fi


   # export_model
   python tools/export_model.py -c $line -o Global.pretrained_model=output/${category}_${model}/RecModel/latest  -o Global.save_inference_dir=./inference/${category}_${model} > $log_path/export_model/${category}_${model}.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m export_model of ${category}_${model}  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/export_model/${category}_${model}.log
      echo -e "\033[31m export_model of ${category}_${model} failed!\033[0m" | tee -a $log_path/result.log
   fi

   # predict
   cd deploy
   case $category in
   Cartoonface)
   python  python/predict_system.py -c configs/inference_cartoon.yaml > ../$log_path/predict/cartoon.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/cartoon.log
   echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
   fi
   ;;
   Logo)
   python  python/predict_system.py -c configs/inference_logo.yaml > ../$log_path/predict/logo.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/logo.log
   echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
   fi
   ;;
   Products)
   python  python/predict_system.py -c configs/inference_product.yaml > ../$log_path/predict/product.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/product.log
   echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
   fi
   ;;
   Vehicle)
   python  python/predict_system.py -c configs/inference_vehicle.yaml > ../$log_path/predict/vehicle.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
   else
      cat ../$log_path/predict/vehicle.log
   echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
   fi
   ;;
   esac
   cd ..
   done
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
