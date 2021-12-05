export FLAGS_cudnn_deterministic=True
cd ${Project_path}

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install --ignore-installed -r requirements.txt

cd examples/csmsc/tts2
source ${PWD}/path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
conf_path=conf/default.yaml
train_output_path=exp/default

# train
sed -i "s/max_epoch: 300/max_epoch: 10/g"  ${conf_path}
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf exp
./local/train.sh ${conf_path} ${train_output_path} > train_2card.log 2>&1
if [ $? -eq 0 ];then
   cat train_2card.log
   echo -e "\033[33m training_2card of speedyspeech successfully! \033[0m"
else
   cat train_2card.log
   echo -e "\033[31m training_2card of speedyspeech failed! \033[0m"
fi
cat train_2card.log | grep "1530/1530" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $6}' > ../../../../log/speedyspeech_baker_2card.log
