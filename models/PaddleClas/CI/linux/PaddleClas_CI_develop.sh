unset GREP_OPTIONS
echo ${cudaid1}
echo ${cudaid2}
echo ${model_flag}
echo ${Data_path}
echo ${Project_path}
echo ${paddle_compile}
export CUDA_VISIBLE_DEVICES=${cudaid2}
export FLAGS_use_virtual_memory_auto_growth=1

echo "path before"
pwd
if [[ ${model_flag} =~ 'CE' ]]; then
   cd ${Project_path}
   echo "path after"
   pwd
   export FLAGS_cudnn_deterministic=True
   unset FLAGS_use_virtual_memory_auto_growth
fi

# <-> model_flag CI是效率云 step0是clas分类 step1是clas分类 step2是clas分类 step3是识别，CI_all是全部都跑
#     pr是TC，clas是分类，rec是识别，single是单独模型debug
# <-> pr_num   随机跑pr的模型数
# <-> python   python版本
# <-> cudaid   cuda卡号
# <-> paddle_compile   paddle包地址
# <-> Data_path   数据路径
#$1 <-> 自己定义的  single_yaml_debug  单独模型yaml字符
#model_clip 根据日期单双数决定剪裁
echo "######  ----ln  data-----"
rm -rf dataset
ln -s ${Data_path} dataset
ls dataset
cd deploy
rm -rf recognition_demo_data_v1.0
rm -rf recognition_demo_data_v1.1
rm -rf models
ln -s ${Data_path}/* .
cd ..

if [[ ${model_flag} =~ 'pr' ]] || [[ ${model_flag} =~ 'single' ]]; then #model_flag
   echo "######  model_flag pr"

   echo "######  ---py37  env -----"
   rm -rf /usr/local/python2.7.15/bin/python
   rm -rf /usr/local/bin/python
   export PATH=/usr/local/bin/python:${PATH}
   case ${python} in #python
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

   unset http_proxy
   unset https_proxy
   echo "######  ----install  paddle-----"
   python -m pip uninstall paddlepaddle-gpu -y
   python -m pip install ${paddle_compile} #paddle_compile

fi

# paddle
echo "######  paddle version"
python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

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
python -m pip install --ignore-installed --upgrade \
   pip -i https://mirror.baidu.com/pypi/simple
python -m pip install  -r requirements.txt  \
   -i https://mirror.baidu.com/pypi/simple
python -m pip install  --ignore-installed paddleslim \
   -i https://mirror.baidu.com/pypi/simple
python -m pip install --ignore-installed dataset/visualdl-2.2.1-py3-none-any.whl \
   -i https://mirror.baidu.com/pypi/simple

rm -rf models_list
rm -rf models_list_all
rm -rf models_list_rec

if [ -d "log" ]; then
   rm -rf log
fi
# dir
log_path=log
phases='train eval infer export_model model_clip predict'
for phase in $phases
do
if [[ -d ${log_path}/${phase} ]]; then
   echo -e "\033[33m ${log_path}/${phase} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${phase}
   echo -e "\033[33m ${log_path}/${phase} is created successfully!\033[0m"
fi
done

#找到diff yaml  &  拆分任务  &  定义要跑的model list
if [[ ${model_flag} =~ 'CE' ]] || [[ ${model_flag} =~ 'CI_step1' ]] || [[ ${model_flag} =~ 'CI_step2' ]] || [[ ${model_flag} =~ 'all' ]] || [[ ${model_flag} =~ 'pr' ]] || [[ ${model_flag} =~ 'clas' ]]; then
   find ppcls/configs/ImageNet/ -name '*.yaml' -exec ls -l {} \; \
      | awk '{print $NF;}'| grep -v 'eval' | grep -v 'kunlun' | grep -v 'fp16' |grep -v 'distill' \
      > models_list_all

   if [[ ${model_flag} =~ 'CI_step0' ]]; then
      cat models_list_all | while read line
      do
      if [[ ${line} =~ 'AlexNet' ]] ||[[ ${line} =~ 'DPN' ]] ||[[ ${line} =~ 'DarkNet' ]] ||[[ ${line} =~ 'DeiT' ]] \
         ||[[ ${line} =~ 'DenseNet' ]] ||[[ ${line} =~ 'EfficientNet' ]] ||[[ ${line} =~ 'GhostNet' ]] \
         ||[[ ${line} =~ 'HRNet' ]] ||[[ ${line} =~ 'HarDNet' ]] ||[[ ${line} =~ 'Inception' ]] \
         ||[[ ${line} =~ 'LeViT' ]] ||[[ ${line} =~ 'MixNet' ]] ||[[ ${line} =~ 'MobileNetV1' ]] \
         ||[[ ${line} =~ 'MobileNetV2' ]] ||[[ ${line} =~ 'MobileNetV3' ]]; then
         echo ${line}  >> models_list
      fi
      done

   elif [[ ${model_flag} =~ 'CI_step1' ]]; then
      cat models_list_all | while read line
      do
      if [[ ${line} =~ 'PPLCNet' ]] || [[ ${line} =~ 'ReXNet' ]] ||[[ ${line} =~ 'RedNet' ]] \
      ||[[ ${line} =~ 'Res2Net' ]] ||[[ ${line} =~ 'ResNeSt' ]] ||[[ ${line} =~ 'ResNeXt' ]]\
      ||[[ ${line} =~ 'ResNeXt101_wsl' ]] ||[[ ${line} =~ 'ResNet' ]] ||[[ ${line} =~ 'SENet' ]]\
      ||[[ ${line} =~ 'ShuffleNet' ]] ||[[ ${line} =~ 'ShuffleNet' ]] ||[[ ${line} =~ 'SqueezeNet' ]]; then
         echo ${line}  >> models_list
      fi
      done

   elif [[ ${model_flag} =~ "pr" ]];then
      shuf -n ${pr_num} models_list_all > models_list

   elif [[ ${model_flag} =~ "clas_single" ]] || [[ ${model_flag} =~ "CE" ]]; then
      echo $1 > models_list

   elif [[ ${model_flag} =~ 'CI_step2' ]]; then
      cat models_list_all | while read line
      do
      if [[ ! ${line} =~ 'AlexNet' ]] && [[ ! ${line} =~ 'DPN' ]] && [[ ! ${line} =~ 'DarkNet' ]] \
      && [[ ! ${line} =~ 'DeiT' ]] && [[ ! ${line} =~ 'DenseNet' ]] && [[ ! ${line} =~ 'EfficientNet' ]] \
      && [[ ! ${line} =~ 'GhostNet' ]] && [[ ! ${line} =~ 'HRNet' ]] && [[ ! ${line} =~ 'HarDNet' ]] \
      && [[ ! ${line} =~ 'Inception' ]] && [[ ! ${line} =~ 'LeViT' ]] && [[ ! ${line} =~ 'MixNet' ]] \
      && [[ ! ${line} =~ 'MobileNetV1' ]] && [[ ! ${line} =~ 'MobileNetV2' ]] && [[ ! ${line} =~ 'MobileNetV3' ]] \
      && [[ ! ${line} =~ 'PPLCNet' ]] && [[ ! ${line} =~ 'ReXNet' ]] && [[ ! ${line} =~ 'RedNet' ]] \
      && [[ ! ${line} =~ 'Res2Net' ]] && [[ ! ${line} =~ 'ResNeSt' ]] && [[ ! ${line} =~ 'ResNeXt' ]]\
      && [[ ! ${line} =~ 'ResNeXt101_wsl' ]] && [[ ! ${line} =~ 'ResNet' ]] && [[ ! ${line} =~ 'SENet' ]]\
      && [[ ! ${line} =~ 'ShuffleNet' ]] && [[ ! ${line} =~ 'SqueezeNet' ]]; then
         echo ${line}  >> models_list
      fi
      done

   elif [[ ${model_flag} =~ 'CI_all' ]] ; then #对应CI_ALL的情况
      shuf models_list_all > models_list

   fi

   echo "######  length models_list"
   wc -l models_list
   cat models_list
   if [[ ${model_flag} =~ "pr" ]];then
      # git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         # | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list
      git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         | grep diff|grep yaml|grep configs|grep ImageNet|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
      echo "######  diff models_list_diff"
      wc -l models_list_diff
      cat models_list_diff
      shuf -n 5 models_list_diff >> models_list #防止diff yaml文件过多导致pr时间过长
   fi
   echo "######  diff models_list"
   wc -l models_list
   cat models_list

   #开始循环
   cat models_list | while read line
   do
   #echo $line
   filename=${line##*/}
   #echo $filename
   model=${filename%.*}
   echo $model

   if [[ ${line} =~ 'fp16' ]];then
      echo "fp16"
      python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
      --upgrade nvidia-dali-cuda102 --ignore-installed -i https://mirror.baidu.com/pypi/simple
   fi

   #visualdl
   echo "######  visualdl "
   ls /root/.visualdl/conf
   rm -rf /root/.visualdl/conf

   if [ -d "output" ]; then
      rm -rf output
   fi
   #train
   #多卡
   if [[ ${model_flag} =~ "CE" ]]; then
      if [[ ${line} =~ 'GoogLeNet' ]] || [[ ${line} =~ 'VGG' ]] || [[ ${line} =~ 'ViT' ]] || [[ ${line} =~ 'PPLCNet' ]] || [[ ${line} =~ 'MobileNetV3' ]]; then
      sed -i 's/learning_rate:/learning_rate: 0.0001 #/g' $line #将 学习率调低为0.0001
      echo "change lr"
      fi
      sed -i 's/RandCropImage/ResizeImage/g' $line
      sed -ie '/RandFlipImage/d' $line
      sed -ie '/flip_code/d' $line
         # -o Global.eval_during_train=False  \
      python -m paddle.distributed.launch tools/train.py -c $line  \
         -o Global.epochs=5  \
         -o Global.seed=1234 \
         -o Global.output_dir=output \
         -o DataLoader.Train.loader.num_workers=0 \
         -o DataLoader.Train.sampler.shuffle=False  \
         -o Global.eval_interval=5  \
         -o Global.save_interval=5 \
         -o DataLoader.Train.sampler.batch_size=4 \
         > $log_path/train/${model}_2card.log 2>&1
   else
      python -m paddle.distributed.launch tools/train.py  \
         -c $line -o Global.epochs=1 \
         -o Global.output_dir=output \
         -o DataLoader.Train.sampler.batch_size=1 \
         -o DataLoader.Eval.sampler.batch_size=1  \
         > $log_path/train/${model}_2card.log 2>&1
   fi
   params_dir=$(ls output)
   echo "######  params_dir"
   echo $params_dir
   if [[ -f "output/$params_dir/latest.pdparams" ]] && [[ $? -eq 0 ]];then
      echo -e "\033[33m training multi of $model  successfully!\033[0m"|tee -a $log_path/result.log
      echo "training_multi_exit_code: 0.0" >> $log_path/train/${model}_2card.log
   else
      cat $log_path/train/${model}_2card.log
      echo -e "\033[31m training multi of $model failed!\033[0m"|tee -a $log_path/result.log
      echo "training_multi_exit_code: 1.0" >> $log_path/train/${model}_2card.log
   fi

   #单卡
   ls output/$params_dir/
   sleep 3
   if [[ ${model_flag} =~ "CE" ]]; then
      rm -rf output #清空多卡cache
      python  tools/train.py -c $line  \
         -o Global.epochs=5  \
         -o Global.seed=1234 \
         -o Global.output_dir=output \
         -o DataLoader.Train.loader.num_workers=0 \
         -o DataLoader.Train.sampler.shuffle=False  \
         -o Global.eval_interval=5  \
         -o Global.save_interval=5 \
         -o DataLoader.Train.sampler.batch_size=4  \
         > $log_path/train/${model}_1card.log 2>&1
   # else  #取消CI单卡训练
   #    python tools/train.py  \
   #       -c $line -o Global.epochs=1 \
   #       -o Global.output_dir=output \
   #       -o DataLoader.Train.sampler.batch_size=1 \
   #       -o DataLoader.Eval.sampler.batch_size=1  \
   #       > $log_path/train/${model}_1card.log 2>&1
      params_dir=$(ls output)
      echo "######  params_dir"
      echo $params_dir
      if [[ -f "output/$params_dir/latest.pdparams" ]] && [[ $? -eq 0 ]];then
         echo -e "\033[33m training single of $model  successfully!\033[0m"|tee -a $log_path/result.log
         echo "training_single_exit_code: 0.0" >> $log_path/train/${model}_1card.log
      else
         cat $log_path/train/${model}_1card.log
         echo -e "\033[31m training single of $model failed!\033[0m"|tee -a $log_path/result.log
         echo "training_single_exit_code: 1.0" >> $log_path/train/${model}_1card.log
      fi
   fi

   if  [[ ${model} =~ 'RedNet' ]] || [[ ${line} =~ 'LeViT' ]] || [[ ${line} =~ 'GhostNet' ]];then
      echo "######  use pretrain model"
      echo ${model}
      wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/${model}_pretrained.pdparams --no-proxy
      rm -rf output/$params_dir/latest.pdparams
      cp -r ${model}_pretrained.pdparams output/$params_dir/latest.pdparams
      rm -rf ${model}_pretrained.pdparams
   fi

   if [[ ${model} =~ 'MobileNetV3' ]] || [[ ${model} =~ 'PPLCNet' ]] \
      || [[ ${line} =~ 'ESNet' ]] || [[ ${line} =~ 'ResNet50.yaml' ]] || [[ ${line} =~ '/ResNet50_vd.yaml' ]];then
      echo "######  use pretrain model"
      echo ${model}
      wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/${model}_pretrained.pdparams --no-proxy
      rm -rf output/$params_dir/latest.pdparams
      cp -r ${model}_pretrained.pdparams output/$params_dir/latest.pdparams
      rm -rf ${model}_pretrained.pdparams
   fi
   sleep 3

   ls output/$params_dir/
   # eval
   python tools/eval.py -c $line \
      -o Global.pretrained_model=output/$params_dir/latest \
      -o DataLoader.Eval.sampler.batch_size=1 \
      > $log_path/eval/$model.log 2>&1
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
      -o Global.pretrained_model=output/$params_dir/latest \
      > $log_path/infer/$model.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
      echo "infer_exit_code: 0.0" >> $log_path/infer/$model.log
   else
      cat $log_path/infer/${model}_infer.log
      echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
      echo "infer_exit_code: 1.0" >> $log_path/infer/$model.log
   fi

   # export_model
   if [[ ${line} =~ 'fp16' ]];then
      python tools/export_model.py -c $line \
         -o Global.pretrained_model=output/$params_dir/latest \
         -o Global.save_inference_dir=./inference/$model \
         -o Arch.data_format="NCHW" \
         > $log_path/export_model/$model.log 2>&1
   else
      python tools/export_model.py -c $line \
         -o Global.pretrained_model=output/$params_dir/latest \
         -o Global.save_inference_dir=./inference/$model \
         > $log_path/export_model/$model.log 2>&1
   fi

   if [ $? -eq 0 ];then
      echo -e "\033[33m export_model of $model  successfully!\033[0m"| tee -a $log_path/result.log
      echo "export_exit_code: 0.0" >> $log_path/export_model/$model.log
   else
      cat $log_path/export_model/$model.log
      echo -e "\033[31m export_model of $model failed!\033[0m" | tee -a $log_path/result.log
      echo "export_exit_code: 1.0" >> $log_path/export_model/$model.log
   fi

   if [[ `expr $RANDOM % 2` -eq 0 ]] && ([[ ${model_flag} =~ 'CI' ]] || [[ ${model_flag} =~ 'single' ]]);then
   # if [[ ${model_flag} =~ 'CI' ]] || [[ ${model_flag} =~ 'single' ]];then #加入随机扰动
      echo "model_clip"
      python model_clip.py --path_prefix="./inference/$model/inference" \
         --output_model_path="./inference/$model/inference" \
         > $log_path/model_clip/$model.log 2>&1
      if [ $? -eq 0 ];then
         echo -e "\033[33m model_clip of $model  successfully!\033[0m"| tee -a $log_path/result.log
      else
         cat $log_path/model_clip/$model.log
         echo -e "\033[31m model_clip of $model failed!\033[0m" | tee -a $log_path/result.log
      fi
   fi

   cd deploy
   if [[ ${model} =~ '384' ]] && [[ ! ${model} =~ 'LeViT' ]];then
      sed -i 's/size: 224/size: 384/g' configs/inference_cls.yaml
      sed -i 's/resize_short: 256/resize_short: 384/g' configs/inference_cls.yaml
      python python/predict_cls.py -c configs/inference_cls.yaml \
         -o Global.inference_model_dir="../inference/"$model \
         > ../$log_path/predict/$model.log 2>&1
      if [ $? -eq 0 ];then
         echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
      else
         cat ../$log_path/predict/${model}.log
         echo -e "\033[31m predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
      fi

      python python/predict_cls.py -c configs/inference_cls.yaml \
         -o Global.infer_imgs="./images"  \
         -o Global.batch_size=4 \
         -o Global.inference_model_dir="../inference/"$model \
         > ../$log_path/predict/$model.log 2>&1
      if [ $? -eq 0 ];then
         echo -e "\033[33m multi_batch_size predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
         echo "predict_exit_code: 0.0" >> ../$log_path/predict/$model.log
      else
         cat ../$log_path/predict/${model}.log
         echo -e "\033[31m multi_batch_size predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
         echo "predict_exit_code: 1.0" >> ../$log_path/predict/$model.log
      fi

      sed -i 's/size: 384/size: 224/g' configs/inference_cls.yaml
      sed -i 's/resize_short: 384/resize_short: 256/g' configs/inference_cls.yaml
   else
      if [[ ${line} =~ 'fp16' ]];then
         python python/predict_cls.py -c configs/inference_cls_ch4.yaml \
            -o Global.inference_model_dir="../inference/"$model \
            > ../$log_path/predict/$model.log 2>&1
      else
         python python/predict_cls.py -c configs/inference_cls.yaml \
            -o Global.inference_model_dir="../inference/"$model \
            > ../$log_path/predict/$model.log 2>&1
      fi

      if [ $? -eq 0 ];then
         echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
      else
         cat ../$log_path/predict/${model}.log
         echo -e "\033[31m predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
      fi

      if [[ ${line} =~ 'fp16' ]];then
      python python/predict_cls.py -c configs/inference_cls_ch4.yaml  \
         -o Global.infer_imgs="./images"  \
         -o Global.batch_size=4 -o Global.inference_model_dir="../inference/"$model \
         > ../$log_path/predict/$model.log 2>&1
      else
      python python/predict_cls.py -c configs/inference_cls.yaml  \
         -o Global.infer_imgs="./images"  \
         -o Global.batch_size=4 \
         -o Global.inference_model_dir="../inference/"$model \
         > ../$log_path/predict/$model.log 2>&1
      fi
      if [ $? -eq 0 ];then
         echo -e "\033[33m multi_batch_size predict of $model  successfully!\033[0m"| tee -a ../$log_path/result.log
         echo "predict_exit_code: 0.0" >> ../$log_path/predict/$model.log
      else
         cat ../$log_path/predict/${model}.log
         echo -e "\033[31m multi_batch_size predict of $model failed!\033[0m"| tee -a ../$log_path/result.log
         echo "predict_exit_code: 1.0" >> ../$log_path/predict/$model.log
      fi

   fi
   cd ..
   done
fi

if [[ ${model_flag} =~ 'CI_step3' ]] || [[ ${model_flag} =~ 'all' ]] || [[ ${model_flag} =~ 'pr' ]] || [[ ${model_flag} =~ 'rec' ]]; then
   echo "######  rec step"
   rm -rf models_list

   find ppcls/configs/Cartoonface/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list_rec
   find ppcls/configs/Logo/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list_rec
   find ppcls/configs/Products/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec
   find ppcls/configs/Vehicle/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec
   find ppcls/configs/slim/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec #后续改成slim
   find ppcls/configs/GeneralRecognition/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list_rec #后续改成slim

   if [[ ${model_flag} =~ 'pr' ]]; then
      shuf -n ${pr_num} models_list_rec > models_list
   elif [[ ${model_flag} =~ "rec_single" ]];then
      echo $1 > models_list
   elif [[ ${model_flag} =~ "CI" ]];then
      shuf models_list_rec > models_list
   fi
   echo "######  rec models_list"
   wc -l models_list
   cat models_list

   if [[ ${model_flag} =~ "pr" ]];then
      # git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         # | grep diff|grep yaml|awk -F 'b/' '{print$2}'|tee -a  models_list
      git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         | grep diff|grep yaml|grep configs|grep Cartoonface|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
      git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         | grep diff|grep yaml|grep configs|grep Logo|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
      git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         | grep diff|grep yaml|grep configs|grep Products|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
      git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         | grep diff|grep yaml|grep configs|grep Vehicle|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
      git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         | grep diff|grep yaml|grep configs|grep slim|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
      git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
         | grep diff|grep yaml|grep configs|grep GeneralRecognition|awk -F 'b/' '{print$2}'|tee -a  models_list_diff_rec
      shuf -n 3 models_list_diff_rec >> models_list #防止diff yaml文件过多导致pr时间过长
      echo "######  diff models_list_diff"
      wc -l models_list_diff_rec
      cat models_list_diff_rec
   fi
   echo "######  rec run models_list"
   wc -l models_list
   cat models_list

   #train
   cat models_list | while read line
   do

   \cp ppcls/data/dataloader/icartoon_dataset.py ppcls/data/dataloader/icartoon_dataset_org.py #保留原始文件
   \cp ppcls/data/dataloader/imagenet_dataset.py ppcls/data/dataloader/imagenet_dataset_org.py
   \cp ppcls/data/dataloader/vehicle_dataset.py ppcls/data/dataloader/vehicle_dataset_org.py

   # if [[ ! ${model_flag} =~ 'CI' ]]; then #全量不修改  #因为时间原因，对于所有数据都进行修改

   # # small data
   # # icartoon_dataset
   sed -ie '/self.images = self.images\[:2000\]/d'  \ppcls/data/dataloader/icartoon_dataset.py
   sed -ie '/self.labels = self.labels\[:2000\]/d'  ppcls/data/dataloader/icartoon_dataset.py
   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.images = self.images\[:2000\]'  ppcls/data/dataloader/icartoon_dataset.py
   sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.labels = self.labels\[:2000\]'  ppcls/data/dataloader/icartoon_dataset.py

   if [[ ${line} =~ 'reid' ]] || ([[ ${line} =~ 'ReID' ]] && [[ ${line} =~ 'Vehicle' ]]) || [[ ${line} =~ 'Logo' ]]; then
      echo "non change vehicle_dataset"
      echo ${line}
   else
      echo "change vehicle_dataset"
      # product_dataset
      sed -ie '/self.images = self.images\[:2000\]/d'  ppcls/data/dataloader/imagenet_dataset.py
      sed -ie '/self.labels = self.labels\[:2000\]/d'  ppcls/data/dataloader/imagenet_dataset.py
      sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.images = self.images\[:2000\]'  ppcls/data/dataloader/imagenet_dataset.py
      sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.labels = self.labels\[:2000\]'  ppcls/data/dataloader/imagenet_dataset.py

      # vehicle_dataset
      sed -ie '/self.images = self.images\[:2000\]/d'  ppcls/data/dataloader/vehicle_dataset.py
      sed -ie '/self.labels = self.labels\[:2000\]/d'  ppcls/data/dataloader/vehicle_dataset.py
      sed -ie '/self.bboxes = self.bboxes\[:2000\]/d'  ppcls/data/dataloader/vehicle_dataset.py
      sed -ie '/self.cameras = self.cameras\[:2000\]/d'  ppcls/data/dataloader/vehicle_dataset.py

      numbers=`grep -n 'assert os.path.exists(self.images\[-1\])' ppcls/data/dataloader/vehicle_dataset.py |awk -F: '{print $1}'`
      number1=`echo $numbers |cut -d' ' -f1`
      sed -i "`echo $number1` a\        self.bboxes = self.bboxes\[:2000\]" ppcls/data/dataloader/vehicle_dataset.py

      numbers=`grep -n 'assert os.path.exists(self.images\[-1\])' ppcls/data/dataloader/vehicle_dataset.py |awk -F: '{print $1}'`
      number2=`echo $numbers |cut -d' ' -f2`
      sed -i "`echo $number2` a\        self.cameras = self.cameras\[:2000\]" ppcls/data/dataloader/vehicle_dataset.py

      sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.images = self.images\[:2000\]'  ppcls/data/dataloader/vehicle_dataset.py
      sed -i '/assert os.path.exists(self.images\[-1\])/a\        self.labels = self.labels\[:2000\]'  ppcls/data/dataloader/vehicle_dataset.py
   fi

   # fi

    #echo $line
    filename=${line##*/}
    #echo $filename
    model=${filename%.*}
    category=`echo $line | awk -F [\/] '{print $3}'`
    echo ${category}_${model}
    echo $category

    if [ -d "output" ]; then
        rm -rf output
    fi
    echo $model

    if [[ ${line} =~ 'Aliproduct' ]]; then
         python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Aliproduct/val_list.txt \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    elif [[ ${line} =~ 'GeneralRecognition' ]]; then
         python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o DataLoader.Train.dataset.image_root=./dataset/Aliproduct/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Aliproduct/val_list.txt \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    elif [[ ${line} =~ 'quantization' ]]; then
         python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Train.sampler.batch_size=32 \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    else
         python -m paddle.distributed.launch tools/train.py  -c $line \
            -o Global.epochs=1 \
            -o Global.save_interval=1 \
            -o Global.eval_interval=1 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o Global.output_dir="./output/"${category}_${model} \
            > $log_path/train/${category}_${model}.log 2>&1
    fi

    params_dir=$(ls output/${category}_${model})
    echo "######  params_dir"
    echo $params_dir

    if [[ $? -eq 0 ]] && [[ $(grep -c -i "Error" $log_path/train/${category}_${model}.log) -eq 0 ]] \
      && [[ -f "output/${category}_${model}/$params_dir/latest.pdparams" ]];then
        echo -e "\033[33m training of ${category}_${model}  successfully!\033[0m"|tee -a $log_path/result.log
    else
        cat $log_path/train/${category}_${model}.log
        echo -e "\033[31m training of ${category}_${model} failed!\033[0m"|tee -a $log_path/result.log
    fi

    # eval
    python tools/eval.py -c $line \
      -o Global.pretrained_model=output/${category}_${model}/$params_dir/latest \
      > $log_path/eval/${category}_${model}.log 2>&1
    if [ $? -eq 0 ];then
        echo -e "\033[33m eval of ${category}_${model}  successfully!\033[0m"| tee -a $log_path/result.log
    else
        cat $log_path/eval/${category}_${model}.log
        echo -e "\033[31m eval of ${category}_${model} failed!\033[0m" | tee -a $log_path/result.log
    fi

    # export_model
    python tools/export_model.py -c $line \
      -o Global.pretrained_model=output/${category}_${model}/$params_dir/latest  \
      -o Global.save_inference_dir=./inference/${category}_${model} \
      > $log_path/export_model/${category}_${model}.log 2>&1
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
    python  python/predict_system.py -c configs/inference_cartoon.yaml \
      > ../$log_path/predict/cartoon.log 2>&1
    if [ $? -eq 0 ];then
        echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
    else
        cat ../$log_path/predict/cartoon.log
    echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
    fi
    ;;
    Logo)
    python  python/predict_system.py -c configs/inference_logo.yaml \
      > ../$log_path/predict/logo.log 2>&1
    if [ $? -eq 0 ];then
        echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
    else
        cat ../$log_path/predict/logo.log
    echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
    fi
    ;;
    Products)
    python  python/predict_system.py -c configs/inference_product.yaml \
      > ../$log_path/predict/product.log 2>&1
    if [ $? -eq 0 ];then
        echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
    else
        cat ../$log_path/predict/product.log
    echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
    fi
    ;;
    Vehicle)
    python  python/predict_system.py -c configs/inference_vehicle.yaml \
      > ../$log_path/predict/vehicle.log 2>&1
    if [ $? -eq 0 ];then
        echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
    else
        cat ../$log_path/predict/vehicle.log
    echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
    fi
    ;;
    esac
    cd ..

   \cp ppcls/data/dataloader/icartoon_dataset_org.py ppcls/data/dataloader/icartoon_dataset.py
   \cp ppcls/data/dataloader/imagenet_dataset_org.py ppcls/data/dataloader/imagenet_dataset.py
   \cp ppcls/data/dataloader/vehicle_dataset_org.py ppcls/data/dataloader/vehicle_dataset.py

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
