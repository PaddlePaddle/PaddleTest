export FLAGS_cudnn_deterministic=True
cd /workspace/PaddleGAN/ce/Paddle_Cloud_CE/src/task/PaddleGAN
sed -i 's/epochs/total_iters/g' configs/firstorder_fashion.yaml #将epcoh换为iter
sed -ie '/- name: PairedRandomHorizontalFlip/d' configs/firstorder_fashion.yaml #删除 - name: PairedRandomHorizontalFlip
sed -ie '/prob: 0.5/{N;d;}' configs/firstorder_fashion.yaml  #删除随机变量 相关参数

rm -rf data
ln -s /home/data/cfs/models_ce/PaddleGAN data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/main.py -c configs/firstorder_fashion.yaml -o total_iters=100 log_config.interval=20 log_config.visiual_interval=999999 snapshot_config.interval=999999 validate.interval=999999 > log/fom_2card.log 2>&1
cat log/fom_2card.log | grep " INFO: Iter: 100/100" > ../log/fom_2card.log
