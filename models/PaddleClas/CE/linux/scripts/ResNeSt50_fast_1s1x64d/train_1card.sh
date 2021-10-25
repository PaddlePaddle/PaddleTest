export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/RandCropImage/ResizeImage/g' ppcls/configs/ImageNet/ResNeSt/ResNeSt50_fast_1s1x64d.yaml
sed -ie '/RandFlipImage/d' ppcls/configs/ImageNet/ResNeSt/ResNeSt50_fast_1s1x64d.yaml
sed -ie '/flip_code/d' ppcls/configs/ImageNet/ResNeSt/ResNeSt50_fast_1s1x64d.yaml

rm -rf dataset
ln -s ${Data_path} dataset
mkdir log
python -m pip install -r requirements.txt
python tools/train.py -c ppcls/configs/ImageNet/ResNeSt/ResNeSt50_fast_1s1x64d.yaml -o Global.epochs=2 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.sampler.batch_size=4 -o DataLoader.Eval.sampler.batch_size=4 > log/ResNeSt50_fast_1s1x64d_1card.log 2>&1
cat log/ResNeSt50_fast_1s1x64d_1card.log | grep Train | grep Avg | grep 'Epoch 2/2' > ../log/ResNeSt50_fast_1s1x64d_1card.log
