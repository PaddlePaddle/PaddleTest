export FLAGS_cudnn_deterministic=True
cd ${Project_path}

sed -i 's!data_lmdb_release/training!data_lmdb_release/validation!g' configs/rec/rec_mv3_none_bilstm_ctc.yml

rm -rf train_data
ln -s ${Data_path}/train_data train_data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt

python tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml -o Global.epoch_num=10 > log/rec_mv3_none_bilstm_ctc_crnn_1card.log 2>&1
cat log/rec_mv3_none_bilstm_ctc_crnn_1card.log | grep "10/10" > ../log/rec_mv3_none_bilstm_ctc_crnn_1card_tmp.log

linenum=`cat ../log/rec_mv3_none_bilstm_ctc_crnn_1card_tmp.log | wc -l`
linenum_last1=`expr $linenum - 1`
if [ $linenum_last1 -eq 0 ]
  then cp ../log/rec_mv3_none_bilstm_ctc_crnn_1card_tmp.log ../log/rec_mv3_none_bilstm_ctc_crnn_1card.log
  else sed ''1,"$linenum_last1"'d' ../log/rec_mv3_none_bilstm_ctc_crnn_1card_tmp.log > ../log/rec_mv3_none_bilstm_ctc_crnn_1card.log
fi
rm -rf ../log/rec_mv3_none_bilstm_ctc_crnn_1card_tmp.log
