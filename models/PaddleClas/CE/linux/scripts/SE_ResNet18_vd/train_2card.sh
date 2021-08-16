export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleClas/ce/Paddle_Cloud_CE/src/task/PaddleClas
sed -i 's/RandCropImage/ResizeImage/g'  ppcls/configs/ImageNet/SENet/SE_ResNet18_vd.yaml
sed -ie '/RandFlipImage/d'  ppcls/configs/ImageNet/SENet/SE_ResNet18_vd.yaml
sed -ie '/flip_code/d'  ppcls/configs/ImageNet/SENet/SE_ResNet18_vd.yaml

rm -rf dataset
ln -s /home/data/cfs/models_ce/PaddleClas dataset
mkdir log
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/train.py -c ppcls/configs/ImageNet/SENet/SE_ResNet18_vd.yaml -o Global.epochs=2 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.sampler.batch_size=4 -o DataLoader.Eval.sampler.batch_size=4 > log/SE_ResNet18_vd_2card.log 2>&1
cat log/SE_ResNet18_vd_1card.log | grep Train | grep Avg | grep 'Epoch 2/2' > ../log/SE_ResNet18_vd_2card.log
