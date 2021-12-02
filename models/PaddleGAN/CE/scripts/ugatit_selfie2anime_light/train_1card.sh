export FLAGS_cudnn_deterministic=True
cd ${Project_path}
sed -i 's/epochs/total_iters/g' configs/ugatit_selfie2anime_light.yaml #将epcoh换为iter
sed -i 's/decay_total_iters/decay_epochs/g' configs/ugatit_selfie2anime_light.yaml #恢复学习率衰减字段
sed -i 's/interval:/interval: 99999 #/g' configs/ugatit_selfie2anime_light.yaml #将epcoh换为iter

rm -rf data
ln -s ${Data_path} data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt
python tools/main.py -c configs/ugatit_selfie2anime_light.yaml -o total_iters=100 log_config.interval=10 > log/ugatit_selfie2anime_light_1card.log 2>&1
cat log/ugatit_selfie2anime_light_1card.log | grep " INFO: Iter: 100/100" > ../log/ugatit_selfie2anime_light_1card.log
