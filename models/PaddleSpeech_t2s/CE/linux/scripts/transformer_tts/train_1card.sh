export FLAGS_cudnn_deterministic=True
cd ${Project_path}

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install --ignore-installed -r requirements.txt

cd examples/ljspeech/tts1
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
conf_path=conf/default.yaml
train_output_path=exp/default
# data preprocess
sed -i "s/python3/python/g" preprocess.sh
rm -rf dump
./local/preprocess.sh ${conf_path} > preprocess.log 2>&1
if [ $? -eq 0 ];then
   cat preprocess.log
   echo -e "\033[33m data preprocess of transformer_tts successfully! \033[0m"
else
   cat preprocess.log
   echo -e "\033[31m data preprocess of transformer_tts failed! \033[0m"
fi

# train
sed -i "s/max_epoch: 500/max_epoch: 1/g;s/batch_size: 16/batch_size: 2/g"  ${conf_path}
sed -i "s/python3/python/g" ./local/train.sh
rm -rf exp
./local/train.sh ${conf_path} ${train_output_path} > train_1card.log 2>&1
if [ $? -eq 0 ];then
   cat train_1card.log
   echo -e "\033[33m training_1card of transformer_tts successfully! \033[0m"
else
   cat train_1card.log
   echo -e "\033[31m training_1card of transformer_tts failed! \033[0m"
fi
sed -i "s/max_epoch: 1/max_epoch: 500/g;s/batch_size: 2/batch_size: 16/g"  ${conf_path}
cat train_1card.log | grep "6450/6450" | awk 'BEGIN{FS=","} {print $9}' > ../../../../log/transformer_tts_1card.log
