export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/lr: 0.1/lr: 0.00001/g' ppcls/configs/ImageNet/VGG/VGG11.yaml #将 学习率调低为0.00001
sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/VGG/VGG11.yaml #将 RandCropImage 字段替换为 ResizeImage
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/VGG/VGG11.yaml #删除 RandFlipImage 字段行
sed -ie '/flip_code/d' ppcls/configs/ImageNet/VGG/VGG11.yaml #删除 flip_code 字段行
rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m pip install -r requirements.txt
python  tools/train.py -c ppcls/configs/ImageNet/VGG/VGG11.yaml  \
    -o Global.epochs=5  \
    -o Global.seed=1234 \
    -o DataLoader.Train.loader.num_workers=0 \
    -o DataLoader.Train.sampler.shuffle=False  \
    -o Global.eval_during_train=False  \
    -o Global.save_interval=5 \
    -o DataLoader.Train.sampler.batch_size=4 > log/vgg11_1card.log 2>&1
cat log/vgg11_1card.log | grep Avg | grep 'Epoch 5/5' > ../log/vgg11_1card.log
