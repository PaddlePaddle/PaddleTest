cd ${Project_path}

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
mkdir log
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install --ignore-installed -r requirements.txt

cd examples/csmsc/voc1
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
conf_path=conf/default.yaml
train_output_path=exp/default

# data preprocess
if [ ! -f "baker_alignment_tone.tar.gz" ]; then
  wget https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz
  tar xf baker_alignment_tone.tar.gz
fi
sed -i "s/python3/python/g" ./local/preprocess.sh
rm -rf dump
./local/preprocess.sh ${conf_path} > preprocess.log 2>&1
if [ $? -eq 0 ];then
   cat preprocess.log
   echo -e "\033[33m data preprocess of parallel wavegan successfully! \033[0m"
else
   cat preprocess.log
   echo -e "\033[31m data preprocess of parallel wavegan failed! \033[0m"
fi

# train
sed -i "s/train_max_steps: 400000/train_max_steps: 500/g;s/save_interval_steps: 5000/save_interval_steps: 500/g;s/eval_interval_steps: 1000/eval_interval_steps: 500/g;s/batch_size: 8/batch_size: 2/g"  ${conf_path}
sed -i "s/python3/python/g" ./local/train.sh
rm -rf exp
./local/train.sh ${conf_path} ${train_output_path} > train_1card.log 2>&1
if [ $? -eq 0 ];then
   cat train_1card.log
   echo -e "\033[33m training_1card of parallel wavegan successfully! \033[0m"
else
   cat train_1card.log
   echo -e "\033[31m training_1card of parallel wavegan failed! \033[0m"
fi
sed -i "s/train_max_steps: 500/train_max_steps: 400000/g;s/save_interval_steps: 500/save_interval_steps: 5000/g;s/eval_interval_steps: 500/eval_interval_steps: 1000/g;s/batch_size: 2/batch_size: 8/g"  ${conf_path}
cat train_1card.log | grep "500/500" | awk 'BEGIN{FS=","} {print $5}' > ../../../../log/parallel_wavegan_baker_1card.log
