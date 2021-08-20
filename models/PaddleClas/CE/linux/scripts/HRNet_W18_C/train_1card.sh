export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleClas/ce/Paddle_Cloud_CE/src/task/PaddleClas
sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml
sed -ie '/flip_code/d' ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml
rm -rf dataset
ln -s /home/data/cfs/models_ce/PaddleClas dataset
mkdir log
python -m pip install -r requirements.txt
python tools/train.py -c ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml  -o Global.epochs=2 -o Global.eval_during_train=False -o DataLoader.Train.sampler.shuffle=False > log/hrnet_1card.log 2>&1
cat log/hrnet_1card.log | grep Avg | grep 'Epoch 2/2' > ../log/hrnet_1card.log
