export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/epochs/total_iters/g' configs/animeganv2.yaml #将epcoh换为iter
sed -i 's/decay_total_iters/decay_epochs/g' configs/animeganv2.yaml #恢复学习率衰减字段
sed -i 's/interval:/interval: 99999 #/g' configs/animeganv2.yaml #将epcoh换为iter
sed -i 's/pretrain_ckpt:/pretrain_ckpt: #/g' configs/animeganv2.yaml

rm -rf data
ln -s ${Data_path} data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python -m paddle.distributed.launch tools/main.py -c configs/animeganv2.yaml -o total_iters=100 log_config.interval=10 > log/animeganv2_2card.log 2>&1
cat log/animeganv2_2card.log | grep " INFO: Iter: 100/100" > ../log/animeganv2_2card.log
