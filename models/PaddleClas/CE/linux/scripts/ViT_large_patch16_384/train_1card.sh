export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/VisionTransformer/ViT_small_patch16_224.yaml
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/VisionTransformer/ViT_small_patch16_224.yaml
sed -ie '/flip_code/d' ppcls/configs/ImageNet/VisionTransformer/ViT_small_patch16_224.yaml
sed -i 's/learning_rate:/learning_rate: 0.0001 #/g' ppcls/configs/ImageNet/VisionTransformer/ViT_small_patch16_224.yaml #将 学习率调低为0.0001

rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m pip install -r requirements.txt
python tools/train.py -c ppcls/configs/ImageNet/VisionTransformer/ViT_small_patch16_224.yaml -o Global.epochs=2 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.sampler.batch_size=4 -o DataLoader.Eval.sampler.batch_size=4 -o Optimizer.lr.learning_rate=0.001 > log/ViT_small_patch16_224_1card.log 2>&1
cat log/ViT_small_patch16_224_1card.log | grep Train | grep Avg | grep 'Epoch 2/2' > ../log/ViT_small_patch16_224_1card.log
