export FLAGS_cudnn_deterministic=True
cd ${Project_path}

sed -i 's!batch_size_per_card: 512!batch_size_per_card: 16!g' configs/cls/cls_mv3.yml
sed -ie '/- RecAug:/{N;d;}' configs/cls/cls_mv3.yml
sed -ie '/- RandAugment:/d' configs/cls/cls_mv3.yml #删除 RandAugment 字段行

rm -rf train_data
ln -s ${Data_path}/train_data train_data
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install -r requirements.txt

python -m paddle.distributed.launch tools/train.py -c configs/cls/cls_mv3.yml -o Global.epoch_num=10 > log/cls_mv3_2card.log 2>&1
cat log/cls_mv3_2card.log | grep "10/10" > ../log/cls_mv3_2card_tmp.log

linenum=`cat ../log/cls_mv3_2card_tmp.log | wc -l`
linenum_last1=`expr $linenum - 1`
if [ $linenum_last1 -eq 0 ]
  then cp ../log/cls_mv3_2card_tmp.log ../log/cls_mv3_2card.log
  else sed ''1,"$linenum_last1"'d' ../log/cls_mv3_2card_tmp.log > ../log/cls_mv3_2card.log
fi
rm -rf ../log/cls_mv3_2card_tmp.log

cat log/cls_mv3_2card.log
