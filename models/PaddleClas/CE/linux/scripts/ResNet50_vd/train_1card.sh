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
python tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml  -o Global.epochs=2 -o Global.eval_during_train=False -o DataLoader.Train.sampler.shuffle=False > log/resnet50_vd_1card.log 2>&1
cat log/resnet50_vd_1card.log | grep Avg | grep 'Epoch 2/2' > ../log/resnet50_vd_1card.log
