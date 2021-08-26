export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleGAN/ce/Paddle_Cloud_CE/src/task/PaddleGAN

if [ ! -f "models/stargan-v2/wing.pdparams" ]; then
  wget -P ./models/stargan-v2/ https://paddlegan.bj.bcebos.com/models/wing.pdparams
fi

sed -i 's/epochs/total_iters/g' configs/starganv2_celeba_hq.yaml #将epcoh换为iter
sed -i 's/decay_total_iters/decay_epochs/g' configs/starganv2_celeba_hq.yaml #恢复学习率衰减字段
sed -i 's/interval:/interval: 99999 #/g' configs/starganv2_celeba_hq.yaml #将epcoh换为iter

rm -rf data
ln -s /home/data/cfs/models_ce/PaddleGAN data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python tools/main.py -c configs/starganv2_celeba_hq.yaml -o total_iters=100 log_config.interval=10 > log/starganv2_celeba_hq_1card.log 2>&1
cat log/starganv2_celeba_hq_1card.log | grep " INFO: Iter: 100/100" > ../log/starganv2_celeba_hq_1card.log
