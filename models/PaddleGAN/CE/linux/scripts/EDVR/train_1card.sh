export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/use_flip: True/use_flip: False/g' configs/edvr_m_wo_tsa.yaml #将 use_flip 字段替换为 Fasle
sed -i 's/use_rot: True/use_rot: False/g' configs/edvr_m_wo_tsa.yaml #将 use_rot 字段替换为 Fasle
rm -rf data
ln -s ${Data_path} data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python tools/main.py -c configs/edvr_m_wo_tsa.yaml -o total_iters=100 dataset.train.use_flip=False dataset.train.use_rot=False  log_config.interval=20 log_config.visiual_interval=999999 snapshot_config.interval=999999 > log/edvr_1card.log 2>&1
cat log/edvr_1card.log | grep " INFO: Iter: 100/100" > ../log/edvr_1card.log
