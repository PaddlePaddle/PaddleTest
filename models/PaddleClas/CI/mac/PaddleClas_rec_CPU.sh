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

# small data
alias sed=gsed
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

# logo_dataset 
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


# pretrained model
ln -s /Users/paddle/PaddleTest/ce_data/PaddleClas/product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams

output_path="output"
cat models_list_rec | while read line
do
#echo $line
filename=${line##*/}
#echo $filename
model=${filename%.*}
category=`echo $line | awk -F [\/] '{print $3}'`
echo ${category}_${model}
echo $category
python tools/train.py  -c $line -o Global.epochs=1 -o Global.save_interval=1 -o Global.eval_interval=1 -o Global.output_dir="./output/"${category}_${model} -o Global.device=cpu
if [ $? -eq 0 ];then
   echo -e "\033[33m training of ${category}_${model}  successfully!\033[0m"|tee -a $log_path/result.log
else
   cat $log_path/train/${category}_${model}.log
   echo -e "\033[31m training of ${category}_${model} failed!\033[0m"|tee -a $log_path/result.log
fi 

# eval
python tools/eval.py -c $line -o Global.pretrained_model=$output_path/${category}_${model}/RecModel/latest -o Global.device=cpu > $log_path/eval/${category}_${model}.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m eval of ${category}_${model}  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/eval/${category}_${model}.log
   echo -e "\033[31m eval of ${category}_${model} failed!\033[0m" | tee -a $log_path/result.log
fi


# export_model
python tools/export_model.py -c $line -o Global.pretrained_model=$output_path/${category}_${model}/RecModel/latest  -o Global.save_inference_dir=./inference/${category}_${model} -o Global.device=cpu > $log_path/export_model/${category}_${model}.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m export_model of ${category}_${model}  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/export_model/${category}_${model}.log
   echo -e "\033[31m export_model of ${category}_${model} failed!\033[0m" | tee -a $log_path/result.log
fi

# predict
cd deploy
ln -s /Users/paddle/PaddleTest/ce_data/PaddleClas/recognition_demo_data_v1.0 recognition_demo_data_v1.0
ln -s /Users/paddle/PaddleTest/ce_data/PaddleClas/models_infer models
rm -rf vector_search/index.so
cp -r /Users/paddle/PaddleTest/scripts/PaddleClas/index.so vector_search/

case $category in
Cartoonface)
  python  python/predict_system.py -c configs/inference_cartoon.yaml -o Global.enable_mkldnn=False  -o Global.use_gpu=False  > ../$log_path/predict/${model}.log 2>&1
  if [ $? -eq 0 ];then
     echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
  else
     cat ../$log_path/predict/${model}.log
  echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
  fi
  ;;
Logo)
  python  python/predict_system.py -c configs/inference_logo.yaml  -o Global.enable_mkldnn=False  -o Global.use_gpu=False  > ../$log_path/predict/${model}.log 2>&1
  if [ $? -eq 0 ];then
     echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
  else
     cat ../$log_path/predict/${model}.log
  echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
  fi
  ;;
Products)
  python  python/predict_system.py -c configs/inference_product.yaml  -o Global.enable_mkldnn=False  -o Global.use_gpu=False  > ../$log_path/predict/${model}.log 2>&1
  if [ $? -eq 0 ];then
     echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
  else
     cat ../$log_path/predict/${model}.log
  echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
  fi
  ;;
Vehicle)
  python  python/predict_system.py -c configs/inference_vehicle.yaml  -o Global.enable_mkldnn=False  -o Global.use_gpu=False > ../$log_path/predict/${model}.log 2>&1
  if [ $? -eq 0 ];then
     echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
  else
     cat ../$log_path/predict/${model}.log
  echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
  fi
  ;;
esac
cd ..
done
