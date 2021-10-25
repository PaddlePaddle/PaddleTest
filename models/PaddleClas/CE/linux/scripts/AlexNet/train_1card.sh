export FLAGS_cudnn_deterministic=True
echo ${Project_path}
echo ${Data_path}
ls;
pwd;
cd ${Project_path}
pwd;

sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/AlexNet/AlexNet.yaml
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/AlexNet/AlexNet.yaml
sed -ie '/flip_code/d' ppcls/configs/ImageNet/AlexNet/AlexNet.yaml

rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m pip install -r requirements.txt
python tools/train.py -c ppcls/configs/ImageNet/AlexNet/AlexNet.yaml -o Global.epochs=2 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.sampler.batch_size=4 -o DataLoader.Eval.sampler.batch_size=4 > log/AlexNet_1card.log 2>&1
cat log/AlexNet_1card.log | grep Train | grep Avg | grep 'Epoch 2/2' > ../log/AlexNet_1card.log
cat log/AlexNet_1card.log
