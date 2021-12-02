export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/epochs/total_iters/g' configs/cyclegan_cityscapes.yaml #将epcoh换为iter
sed -i 's/decay_total_iters/decay_epochs/g' configs/cyclegan_cityscapes.yaml #恢复学习率衰减字段
sed -i 's/RandomCrop/Resize/g' configs/cyclegan_cityscapes.yaml #将 RandomCrop 字段替换为 Resize
sed -ie '/- name: RandomHorizontalFlip/d' configs/cyclegan_cityscapes.yaml #删除 RandomHorizontalFlip 字段行
sed -ie '/prob: 0.5/d' configs/cyclegan_cityscapes.yaml #删除 prob 字段行

rm -rf data
ln -s ${Data_path} data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt

python tools/main.py -c configs/cyclegan_cityscapes.yaml -o total_iters=100 log_config.interval=20 log_config.visiual_interval=999999 snapshot_config.interval=999999 > log/cyclegan_1card.log 2>&1
cat log/cyclegan_1card.log | grep " INFO: Iter: 100/100" > ../log/cyclegan_1card.log
