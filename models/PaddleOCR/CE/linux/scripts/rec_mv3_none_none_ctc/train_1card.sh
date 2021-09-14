export FLAGS_cudnn_deterministic=True
cd ${Project_path}
# data
rm -rf train_data
ln -s ${Data_path}/train_data train_data
# log
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
# depencency
python -m pip install -r requirements.txt
# small data
sed -i 's!data_lmdb_release/training!data_lmdb_release/validation!g' configs/rec/rec_mv3_none_none_ctc.yml

python tools/train.py -c configs/rec/rec_mv3_none_none_ctc.yml -o Global.epoch_num=10 > log/rec_mv3_none_none_ctc_1card.log 2>&1
cat log/rec_mv3_none_none_ctc_1card.log | grep "10/10" > ../log/rec_mv3_none_none_ctc_1card.log
