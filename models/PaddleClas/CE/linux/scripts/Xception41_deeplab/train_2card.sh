export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/RandCropImage/ResizeImage/g'  ppcls/configs/ImageNet/Xception/Xception41_deeplab.yaml
sed -ie '/RandFlipImage/d'  ppcls/configs/ImageNet/Xception/Xception41_deeplab.yaml
sed -ie '/flip_code/d'  ppcls/configs/ImageNet/Xception/Xception41_deeplab.yaml

rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/train.py -c ppcls/configs/ImageNet/Xception/Xception41_deeplab.yaml -o Global.epochs=2 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.sampler.batch_size=4 -o DataLoader.Eval.sampler.batch_size=4 > log/Xception41_deeplab_2card.log 2>&1
cat log/Xception41_deeplab_1card.log | grep Train | grep Avg | grep 'Epoch 2/2' > ../log/Xception41_deeplab_2card.log
