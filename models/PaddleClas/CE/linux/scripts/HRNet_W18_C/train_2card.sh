export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml
sed -ie '/flip_code/d' ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml

rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml  -o Global.epochs=2 -o Global.eval_during_train=False -o DataLoader.Train.sampler.shuffle=False > log/hrnet_2card.log 2>&1
cat log/hrnet_2card.log | grep Avg | grep 'Epoch 2/2' > ../log/hrnet_2card.log
