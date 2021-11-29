export FLAGS_cudnn_deterministic=True
cd ${Project_path}

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install --ignore-installed -r requirements.txt

cd examples/ljspeech/voc0
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
preprocess_path=preprocessed_ljspeech
train_output_path=output
# train
rm -rf output
python ${BIN_DIR}/train.py --data=${preprocess_path} --output=${train_output_path} --ngpu=2 --opts data.batch_size 2 training.max_iteration 500 training.valid_interval 500 training.save_interval 500 > train_2card.log 2>&1
if [ $? -eq 0 ];then
   cat train_2card.log
   echo -e "\033[33m training_2card of waveflow successfully! \033[0m"
else
   cat train_2card.log
   echo -e "\033[31m training_2card of waveflow failed! \033[0m"
fi
cat train_2card.log | grep "step: 500" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $4}' > ./tmp_2card.log
sed -i "s/-//g" ./tmp_2card.log
cat tmp_2card.log > ../../../../log/waveflow_2card.log
