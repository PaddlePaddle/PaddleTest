export FLAGS_cudnn_deterministic=True
cd ${Project_path}

line=ppcls/configs/ImageNet/DPN/DPN107.yaml
filename=${line##*/}
#echo $filename
model=${filename%.*}
echo $model

sed -i 's/RandCropImage/ResizeImage/g' $line
sed -ie '/RandFlipImage/d' $line
sed -ie '/flip_code/d' $line

if [ -d "output" ]; then
   rm -rf output
else
   python -m pip install -r requirements.txt
fi

rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m paddle.distributed.launch tools/train.py -c $line  \
    -o Global.epochs=5  \
    -o Global.seed=1234 \
    -o Global.output_dir=output \
    -o DataLoader.Train.loader.num_workers=0 \
    -o DataLoader.Train.sampler.shuffle=False  \
    -o Global.eval_during_train=False  \
    -o Global.save_interval=5 \
    -o DataLoader.Train.sampler.batch_size=4 > log/${model}_2card.log 2>&1
cat log/${model}_2card.log | grep Train | grep Avg | grep 'Epoch 5/5' > ../log/${model}_2card.log

params_dir=$(ls output)
echo "######  params_dir"
echo $params_dir

if [ -f "output/$params_dir/latest.pdparams" ];then
   echo -e "\033[33m training of ${model}  successfully!\033[0m"
else
   cat log/${model}_2card.log
   echo -e "\033[31m training of ${model} failed!\033[0m"
fi