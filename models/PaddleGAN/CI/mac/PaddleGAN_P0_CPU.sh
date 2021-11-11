unset GREP_OPTIONS
# data
rm -rf data
ln -s ${data_path} data

# env
python -m pip install --upgrade pip -i
python -m pip install -r requirements.txt
python -m pip install -v -e .
python -m pip install dlib


sed -i '' 's/epochs/total_iters/g' configs/cyclegan_cityscapes.yaml #将epcoh换为iter
sed -i '' 's/decay_total_iters/decay_epochs/g' configs/cyclegan_cityscapes.yaml #恢复学习率衰减字段
sed -i '' 's/RandomCrop/Resize/g' configs/cyclegan_cityscapes.yaml #将 RandomCrop 字段替换为 Resize
sed -ie '/- name: RandomHorizontalFlip/d' configs/cyclegan_cityscapes.yaml #删除 RandomHorizontalFlip 字段行
sed -ie '/prob: 0.5/d' configs/cyclegan_cityscapes.yaml #删除 prob 字段行

sed -ie '/- name: PairedRandomHorizontalFlip/{N;d;}' configs/esrgan_x4_div2k.yaml  #删除随机变量
sed -ie '/- name: PairedRandomVerticalFlip/{N;d;}' configs/esrgan_x4_div2k.yaml
sed -ie '/- name: PairedRandomTransposeHW/{N;d;}' configs/esrgan_x4_div2k.yaml

sed -i '' 's/use_flip: True/use_flip: False/g' configs/edvr_m_wo_tsa.yaml #将 use_flip 字段替换为 Fasle
sed -i '' 's/use_rot: True/use_rot: False/g' configs/edvr_m_wo_tsa.yaml #将 use_rot 字段替换为 Fasle

sed -i '' 's/epochs/total_iters/g' configs/firstorder_fashion.yaml #将epcoh换为iter
sed -i '' 's/decay_total_iters/decay_epochs/g' configs/firstorder_fashion.yaml #恢复学习率衰减字段
sed -ie '/- name: PairedRandomHorizontalFlip/d' configs/firstorder_fashion.yaml #删除 - name: PairedRandomHorizontalFlip
sed -ie '/prob: 0.5/{N;d;}' configs/firstorder_fashion.yaml  #删除随机变量 相关参数

# dir
log_path=log
phases='train eval infer'


for phase in $phases
do
if [[ -d ${log_path}/${phase} ]]; then
   echo -e "\033[33m ${log_path}/${phase} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${phase}
   echo -e "\033[33m ${log_path}/${phase} is created successfully!\033[0m"
fi
done

cat gan_models_list_P0 | while read line
do
#echo $line
filename=${line##*/}
#echo $filename
model=${filename%.*}
echo $model
if [ -d "output" ]; then
   rm -rf output
fi
# sed -i '' 's/epochs/total_iters/g' $line #将epcoh换为iter
# sed -i '' 's/decay_total_iters/decay_epochs/g' $line #恢复学习率衰减字段
#train
python tools/main.py -c $line -o total_iters=20 log_config.interval=20 log_config.visiual_interval=1 snapshot_config.interval=10 output_dir=output > $log_path/train/$model.log 2>&1
params_dir=$(ls output)
echo "params_dir"
if [ -f "output/$params_dir/iter_20_checkpoint.pdparams" ];then
   echo -e "\033[33m training of $model  successfully!\033[0m"|tee -a $log_path/result.log
else
   cat $log_path/train/$model.log
   echo -e "\033[31m training of $model failed!\033[0m"|tee -a $log_path/result.log
fi
sleep 5

if [[ ! ${line} =~ 'firstorder' ]]; then
   #eval
   python tools/main.py -c $line --evaluate-only --load output/$params_dir/iter_20_checkpoint.pdparams > $log_path/eval/${model}.log 2>&1
   if [[ $? -eq 0 ]];then
      echo -e "\033[33m evaluate of $model  successfully!\033[0m"| tee -a $log_path/result.log
   else
      cat $log_path/eval/${model}.log
      echo -e "\033[31m evaluate of $model failed!\033[0m"| tee -a $log_path/result.log
   fi
fi
done

#infer
python -u applications/tools/styleganv2.py --output_path styleganv2_infer --model_type ffhq-config-f --seed 233 --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --n_row 3 --n_col 5 > $log_path/infer/styleganv2.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of styleganv2  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/styleganv2.log
   echo -e "\033[31m infer of styleganv2 failed!\033[0m"| tee -a $log_path/result.log
fi
# Wav2Lip
# python applications/tools/wav2lip.py --face ./docs/imgs/mona7s.mp4 --audio ./docs/imgs/guangquan.m4a --outfile Wav2Lip_infer.mp4 > $log_path/infer/wav2lip.log 2>&1
# if [[ $? -eq 0 ]];then
#    echo -e "\033[33m infer of wav2lip  successfully!\033[0m"| tee -a $log_path/result.log
# else
#    cat $log_path/infer/wav2lip.log
#    echo -e "\033[31m infer of wav2lip failed!\033[0m"| tee -a $log_path/result.log
# fi
# animeganv2
python applications/tools/animeganv2.py --input_image ./docs/imgs/pSp-input.jpg > $log_path/infer/animeganv2.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of animeganv2  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/animeganv2.log
   echo -e "\033[31m infer of animeganv2 failed!\033[0m"| tee -a $log_path/result.log
fi
# fist order motion model
python -u applications/tools/first-order-demo.py --driving_video ./docs/imgs/fom_dv.mp4 --source_image ./docs/imgs/fom_source_image.png --ratio 0.4 --relative --adapt_scale > $log_path/infer/fist_order_motion_model.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of fist order motion model  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/fist_order_motion_model.log
   echo -e "\033[31m infer of fist order motion model failed!\033[0m"| tee -a $log_path/result.log
fi
# # fist order motion model multi_person
# python -u applications/tools/first-order-demo.py --driving_video ./docs/imgs/fom_dv.mp4 --source_image ./docs/imgs/fom_source_image_multi_person.jpg --ratio 0.4 --relative --adapt_scale --multi_person > $log_path/infer/fist_order_motion_model_multi_person.log 2>&1
# if [[ $? -eq 0 ]];then
#    echo -e "\033[33m infer of fist order motion model  multi_person successfully!\033[0m"| tee -a $log_path/result.log
# else
#    cat $log_path/infer/fist_order_motion_model.log
#    echo -e "\033[31m infer of fist order motion model multi_person failed!\033[0m"| tee -a $log_path/result.log
# fi

# face_parse
python applications/tools/face_parse.py --input_image ./docs/imgs/face.png > $log_path/infer/face_parse.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of face_parse  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/face_parse.log
   echo -e "\033[31m infer of face_parse failed!\033[0m"| tee -a $log_path/result.log
fi
# psgan
python tools/psgan_infer.py --config-file configs/makeup.yaml --source_path  docs/imgs/ps_source.png --reference_dir docs/imgs/ref --evaluate-only > $log_path/infer/psgan.log 2>&1
if [[ $? -eq 0 ]];then
   echo -e "\033[33m infer of psgan  successfully!\033[0m"| tee -a $log_path/result.log
else
   cat $log_path/infer/psgan.log
   echo -e "\033[31m infer of psgan failed!\033[0m"| tee -a $log_path/result.log
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
