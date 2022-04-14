export FLAGS_cudnn_deterministic=True
cd ${Project_path}

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
if [ ! -d "../log" ]; then
  mkdir ../log
fi

cd examples/csmsc/tts0
source ${PWD}/path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
conf_path=conf/default.yaml
train_output_path=exp/default
# data preprocess
if [ ! -f "baker_alignment_tone.tar.gz" ]; then
   wget https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz
   tar -xf baker_alignment_tone.tar.gz
fi
sed -i "s/python3/python/g" ./local/preprocess.sh
rm -rf dump
./local/preprocess.sh ${conf_path} > preprocess.log 2>&1
if [ $? -eq 0 ];then
   cat preprocess.log
   echo -e "\033[33m data preprocess of tacotron2 successfully! \033[0m"
else
   cat preprocess.log
   echo -e "\033[31m data preprocess of tacotron2 failed! \033[0m"
fi

# train
sed -i "s/max_epoch: 200/max_epoch: 5/g;s/batch_size: 64/batch_size: 32/g" ${conf_path}
cat ${conf_path}
sed -i "s/python3/python/g" ./local/train.sh
rm -rf exp
./local/train.sh ${conf_path} ${train_output_path} > train_1card.log 2>&1
if [ $? -eq 0 ];then
   cat train_1card.log
   echo -e "\033[33m training_1card of tacotron2 successfully! \033[0m"
else
   cat train_1card.log
   echo -e "\033[31m training_1card of tacotron2 failed! \033[0m"
fi
sed -i "s/max_epoch: 5/max_epoch: 200/g" ${conf_path}
cat train_1card.log | grep "1530/1530" | grep "Rank: 0" | awk 'BEGIN{FS=","} {print $7}' > ../../../../log/tacotron2_1card.log
