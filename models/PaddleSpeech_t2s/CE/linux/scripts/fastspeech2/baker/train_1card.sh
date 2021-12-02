export FLAGS_cudnn_deterministic=True
cd ${Project_path}

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
ls ~/datasets
if [ ! -d "../log" ]; then
  mkdir ../log
fi
python -m pip install --ignore-installed -r requirements.txt

cd examples/csmsc/tts3
pwd
ls

# export MAIN_ROOT=`realpath ${PWD}/../../../`
# export PATH=${MAIN_ROOT}:${MAIN_ROOT}/utils:${PATH}
# export LC_ALL=C
# export PYTHONDONTWRITEBYTECODE=1
# # Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
# export PYTHONIOENCODING=UTF-8
# export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}
# MODEL=fastspeech2
# export BIN_DIR=${MAIN_ROOT}/paddlespeech/t2s/exps/${MODEL}
source ${PWD}/path.sh
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
bash ./local/preprocess.sh ${conf_path} > preprocess.log 2>&1
if [ $? -eq 0 ];then
   cat preprocess.log
   echo -e "\033[33m data preprocess of fastspeech2_baker successfully! \033[0m"
else
   cat preprocess.log
   echo -e "\033[31m data preprocess of fastspeech2_baker failed! \033[0m"
fi

# train
sed -i "s/max_epoch: 1000/max_epoch: 5/g;s/batch_size: 64/batch_size: 16/g" ${conf_path}
sed -i "s/python3/python/g" ./local/train.sh
rm -rf exp
./local/train.sh ${conf_path} ${train_output_path} > train_1card.log 2>&1
if [ $? -eq 0 ];then
   cat train_1card.log
   echo -e "\033[33m training_1card of fastspeech2_baker successfully! \033[0m"
else
   cat train_1card.log
   echo -e "\033[31m training_1card of fastspeech2_baker failed! \033[0m"
fi
sed -i "s/max_epoch: 5/max_epoch: 1000/g;s/batch_size: 16/batch_size: 64/g" ${conf_path}
cat train_1card.log | grep "3060/3060" | awk 'BEGIN{FS=","} {print $7}' > ../../../../log/fastspeech2_baker_1card.log
