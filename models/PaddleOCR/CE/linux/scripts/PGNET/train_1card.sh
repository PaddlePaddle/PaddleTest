export FLAGS_cudnn_deterministic=True
cd ${Project_path}

sed -i 's!batch_size_per_card: 14!batch_size_per_card: 4!g' configs/e2e/e2e_r50_vd_pg.yml

rm -rf train_data
ln -s ${Data_path}/train_data train_data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt

python tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml  -o Global.epoch_num=10 > log/e2e_r50_vd_pgnet_1card.log 2>&1
cat log/e2e_r50_vd_pgnet_1card.log 
cat log/e2e_r50_vd_pgnet_1card.log | grep "10/10" > ../log/e2e_r50_vd_pgnet_1card.log
