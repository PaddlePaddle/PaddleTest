echo "Project_path"
echo ${Project_path}
cd ${Project_path}
echo "path after"
pwd
# env
unset http_proxy
unset https_proxy
python -m pip install -r requirements.txt --ignore-installed  -i https://pypi.tuna.tsinghua.edu.cn/simple

# dir
log_path=log
gpu_flag=False


phases='train eval infer export predict'
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
cp ../ocr_p0model_list ./
cat ocr_p0model_list | while read line
do
#echo $line
filename=${line##*/}
#echo $filename
model=${filename%.*}
echo $model

if [[ $model =~ "cls" ]];then
   category="cls"
elif [[ $model =~ "det" ]];then
   category="det"
elif [[ $model =~ "rec" ]];then
   category="rec"
elif [[ $model =~ "e2e" ]];then
   category="e2e"
fi
echo "category"
echo $category

#train
python tools/train.py -c $line  -o Global.epoch_num=1 Global.use_gpu=${gpu_flag} Global.save_epoch_step=1 Global.save_model_dir="output/"${model} > $log_path/train/$model.log 2>&1
if [[ $? -eq 0 ]]  && [[ $(grep -c "Error" $log_path/train/$model.log) -eq 0 ]] && [ -f "output/"${model}"/latest.pdparams" ];then
   echo -e "\033[33m training of $model  successfully!\033[0m"|tee -a $log_path/result.log
   echo "training_exit_code: 0.0" >> $log_path/train/$model.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m"|tee -a $log_path/result.log
   echo "training_exit_code: 1.0" >> $log_path/train/$model.log
fi

#eval
if [[ $model =~ "e2e" ]];then
python tools/eval.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="pretrain_models/en_server_pgnetA/best_accuracy.pdparams" > $log_path/eval/$model.log 2>&1
else
python tools/eval.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest.pdparams" > $log_path/eval/$model.log 2>&1
fi
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/eval/$model.log) -eq 0 ]];then
   echo -e "\033[33m eval of $model  successfully!\033[0m" | tee -a $log_path/result.log
   echo "eval_exit_code: 0.0" >> $log_path/eval/$model.log
else
   cat $log_path/eval/$model.log
   echo -e "\033[31m eval of $model failed!\033[0m" | tee -a $log_path/result.log
   echo "eval_exit_code: 1.0" >> $log_path/eval/$model.log
fi

#infer
echo "infer"
if [[ $model =~ "cls" ]];then
echo "cls"

python tools/infer_${category}.py -c $line -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest"  Global.infer_img="./doc/imgs_words/ch/word_1.jpg" > $log_path/infer/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/infer/${model}.log) -eq 0 ]];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 0.0" >> $log_path/infer/$model.log
else
   cat $log_path/infer/${model}.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 1.0" >> $log_path/infer/$model.log
fi

elif [[ $model =~ "det" ]];then
echo "det"

python tools/infer_${category}.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest" Global.infer_img="./doc/imgs_en/img_10.jpg" > $log_path/infer/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/infer/${model}.log) -eq 0 ]];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 0.0" >> $log_path/infer/$model.log
else
   cat $log_path/infer/${model}.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 1.0" >> $log_path/infer/$model.log
fi

elif [[ $model =~ "rec" ]];then
echo "rec"

python tools/infer_${category}.py -c $line  -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest" Global.infer_img="./doc/imgs_words/en/word_1.png" Global.test_batch_size_per_card=1 > $log_path/infer/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/infer/${model}.log) -eq 0 ]];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 0.0" >> $log_path/infer/$model.log
else
   cat $log_path/infer/${model}.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 1.0" >> $log_path/infer/$model.log
fi

elif [[ $model =~ "e2e" ]];then
echo "e2e"

python tools/infer_${category}.py -c $line -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest" Global.infer_img="./doc/imgs_en/img_10.jpg"  > $log_path/infer/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/infer/${model}.log) -eq 0 ]];then
   echo -e "\033[33m infer of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 0.0" >> $log_path/infer/$model.log
else
   cat $log_path/infer/${model}.log
   echo -e "\033[31m infer of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "infer_exit_code: 1.0" >> $log_path/infer/$model.log
fi
fi

#export
echo "export"
python tools/export_model.py -c $line -o Global.use_gpu=${gpu_flag} Global.checkpoints="output/"${model}"/latest"  Global.save_inference_dir="./models_inference/"${model} >  $log_path/export/${model}.log 2>&1
if [[ $? -eq 0 ]] && [[ $(grep -c "Error" $log_path/export/${model}.log) -eq 0 ]];then
   echo -e "\033[33m export of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "export_exit_code: 0.0" >> $log_path/export/$model.log
else
   cat $log_path/export/${model}.log
   echo -e "\033[31m export of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "export_exit_code: 1.0" >> $log_path/export/$model.log
fi

#predict
echo "predict"

if [[ $model =~ "cls" ]];then
echo "cls"

python tools/infer/predict_${category}.py --image_dir="./doc/imgs_words/ch/word_4.jpg" --cls_model_dir="./models_inference/"${model} --use_gpu=${gpu_flag} >$log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 0.0" >> $log_path/predict/$model.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 1.0" >> $log_path/predict/$model.log
fi

elif [[ $model =~ "det" ]];then
echo "det"

python tools/infer/predict_${category}.py  --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./models_inference/"${model} --use_gpu=${gpu_flag} >$log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 0.0" >> $log_path/predict/$model.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 1.0" >> $log_path/predict/$model.log
fi

elif [[ $model =~ "rec" ]];then
echo "rec"

#python tools/infer/predict_${category}.py  --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_type="en" --use_gpu=${gpu_flag} >$log_path/predict/${model}.log 2>&1
python tools/infer/predict_${category}.py  --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"${model} --rec_image_shape="3, 32, 100" --rec_char_dict_path=./ppocr/utils/ic15_dict.txt --use_gpu=${gpu_flag} >$log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 0.0" >> $log_path/predict/$model.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 1.0" >> $log_path/predict/$model.log
fi

elif [[ $model =~ "e2e" ]];then
echo "e2e"

#python tools/infer/predict_${category}.py   --image_dir="./doc/imgs_en/img_10.jpg"  --e2e_model_dir="./models_inference/"${model} --e2e_algorithm="PGNet" --e2e_pgnet_polygon=False --use_gpu=${gpu_flag} >$log_path/predict/${model}.log 2>&1
python tools/infer/predict_${category}.py   --image_dir="./doc/imgs_en/img_10.jpg"  --e2e_model_dir="./models_inference/"${model} --e2e_algorithm="PGNet" --use_gpu=${gpu_flag} >$log_path/predict/${model}.log 2>&1
if [[ $? -eq 0 ]]; then
   echo -e "\033[33m predict of $model  successfully!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 0.0" >> $log_path/predict/$model.log
else
   cat $log_path/predict/${model}.log
   echo -e "\033[31m predict of $model failed!\033[0m"| tee -a $log_path/result.log
   echo "predict_exit_code: 1.0" >> $log_path/predict/$model.log
fi

fi
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
