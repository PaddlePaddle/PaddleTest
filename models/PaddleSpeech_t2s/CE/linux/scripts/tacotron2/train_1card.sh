export FLAGS_cudnn_deterministic=True
cd ${Project_path}

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install --ignore-installed -r requirements.txt

cd examples/ljspeech/tts0
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
preprocess_path=preprocessed_ljspeech
train_output_path=output
# data preprocess
sed -i "s/python3/python/g" ./local/preprocess.sh
rm -rf preprocessed_ljspeech
./local/preprocess.sh ${preprocess_path} > preprocess.log 2>&1
if [ $? -eq 0 ];then
   cat preprocess.log
   echo -e "\033[33m data preprocess of tacotron2 successfully! \033[0m"
else
   cat preprocess.log
   echo -e "\033[31m data preprocess of tacotron2 failed! \033[0m"
fi

# train
rm -rf output
python ${BIN_DIR}/train.py --data=${preprocess_path} --output=${train_output_path} --ngpu=1 --opts data.batch_size 2 training.max_iteration 500 training.valid_interval 500 training.save_interval 500 > train_1card.log 2>&1
if [ $? -eq 0 ];then
   cat train_1card.log
   echo -e "\033[33m training_1card of tacotron2 successfully! \033[0m"
else
   cat train_1card.log
   echo -e "\033[31m training_1card of tacotron2 failed! \033[0m"
fi
cat train_1card.log | grep "step: 500" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $4}' > ../../../../log/tacotron2_1card.log
