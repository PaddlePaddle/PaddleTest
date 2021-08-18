export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleGAN/ce/Paddle_Cloud_CE/src/task/PaddleGAN
sed -ie '/- name: PairedRandomHorizontalFlip/{N;d;}' configs/esrgan_x4_div2k.yaml  #删除随机变量
sed -ie '/- name: PairedRandomVerticalFlip/{N;d;}' configs/esrgan_x4_div2k.yaml
sed -ie '/- name: PairedRandomTransposeHW/{N;d;}' configs/esrgan_x4_div2k.yaml
rm -rf data
ln -s /home/data/cfs/models_ce/PaddleGAN data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/main.py -c configs/esrgan_x4_div2k.yaml -o total_iters=100 log_config.interval=20 log_config.visiual_interval=999999 snapshot_config.interval=999999 dataset.train.batch_size=1 > log/esrgan_2card.log 2>&1
cat log/esrgan_2card.log | grep " INFO: Iter: 100/100" > ../log/esrgan_2card.log
