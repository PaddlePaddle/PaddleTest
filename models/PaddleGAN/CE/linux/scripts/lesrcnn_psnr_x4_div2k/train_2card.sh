export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleGAN/ce/Paddle_Cloud_CE/src/task/PaddleGAN
sed -i 's/epochs/total_iters/g' configs/lesrcnn_psnr_x4_div2k.yaml #将epcoh换为iter
sed -i 's/decay_total_iters/decay_epochs/g' configs/lesrcnn_psnr_x4_div2k.yaml #恢复学习率衰减字段
sed -i 's/interval:/interval: 99999 #/g' configs/lesrcnn_psnr_x4_div2k.yaml #将epcoh换为iter

rm -rf data
ln -s /home/data/cfs/models_ce/PaddleGAN data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/main.py -c configs/lesrcnn_psnr_x4_div2k.yaml -o total_iters=100 log_config.interval=10 > log/lesrcnn_psnr_x4_div2k_2card.log 2>&1
cat log/lesrcnn_psnr_x4_div2k_2card.log | grep " INFO: Iter: 100/100" > ../log/lesrcnn_psnr_x4_div2k_2card.log
