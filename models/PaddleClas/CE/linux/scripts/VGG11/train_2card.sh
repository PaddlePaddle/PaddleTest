export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleClas/ce/Paddle_Cloud_CE/src/task/PaddleClas
sed -i 's/lr: 0.1/lr: 0.00001/g' ppcls/configs/ImageNet/VGG/VGG11.yaml #将 学习率调低为0.00001
sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/VGG/VGG11.yaml #将 RandCropImage 字段替换为 ResizeImage
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/VGG/VGG11.yaml #删除 RandFlipImage 字段行
sed -ie '/flip_code/d' ppcls/configs/ImageNet/VGG/VGG11.yaml #删除 flip_code 字段行
rm -rf dataset
ln -s /home/data/cfs/models_ce/PaddleClas dataset
mkdir log
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/train.py -c ppcls/configs/ImageNet/VGG/VGG11.yaml -o Global.epochs=2 -o Global.eval_during_train=False > log/vgg11_2card.log 2>&1
cat log/vgg11_2card.log | grep Avg | grep 'Epoch 2/2' > ../log/vgg11_2card.log

