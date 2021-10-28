export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml
sed -ie '/flip_code/d' ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml
# sed -i '46,47d' ppcls/configs/ImageNet/g/ResNet/ResNet50_vd.yaml
rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m pip install -r requirements.txt
python  tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml  \
    -o Global.epochs=5  \
    -o Global.seed=1234 \
    -o DataLoader.Train.loader.num_workers=0 \
    -o DataLoader.Train.sampler.shuffle=False  \
    -o Global.eval_during_train=False  \
    -o Global.save_interval=5 \
    -o DataLoader.Train.sampler.batch_size=4 > log/resnet50_vd_1card.log 2>&1
cat log/resnet50_vd_1card.log | grep Avg | grep 'Epoch 5/5' > ../log/resnet50_vd_1card.log
