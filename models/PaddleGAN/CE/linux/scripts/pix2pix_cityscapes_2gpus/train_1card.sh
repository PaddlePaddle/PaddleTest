export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleGAN/ce/Paddle_Cloud_CE/src/task/PaddleGAN
sed -i 's/epochs/total_iters/g' configs/pix2pix_cityscapes_2gpus.yaml #将epcoh换为iter
sed -i 's/decay_total_iters/decay_epochs/g' configs/pix2pix_cityscapes_2gpus.yaml #恢复学习率衰减字段
sed -i 's/interval:/interval: 99999 #/g' configs/pix2pix_cityscapes_2gpus.yaml #将epcoh换为iter

rm -rf data
ln -s /home/data/cfs/models_ce/PaddleGAN data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python tools/main.py -c configs/pix2pix_cityscapes_2gpus.yaml -o total_iters=100 log_config.interval=10 > log/pix2pix_cityscapes_2gpus_1card.log 2>&1
cat log/pix2pix_cityscapes_2gpus_1card.log | grep " INFO: Iter: 100/100" > ../log/pix2pix_cityscapes_2gpus_1card.log
