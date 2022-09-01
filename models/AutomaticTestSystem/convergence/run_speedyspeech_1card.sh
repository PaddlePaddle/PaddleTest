Data_path=/ssd2/ce_data/PaddleSpeech_t2s

cd PaddleSpeech
# python -m pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
# python -m pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple


python -m pip install praatio
apt-get update
apt-get install libsndfile1

rm -rf ~/datasets
ln -s ${Data_path}/train_data ~/datasets
cd examples/csmsc/tts2
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
# rm -rf dump
# bash ./local/preprocess.sh ${conf_path} > preprocess.log 2>&1
if [ $? -eq 0 ];then
   cat preprocess.log
   echo -e "\033[33m data preprocess of fastspeech2_baker successfully! \033[0m"
else
   cat preprocess.log
   rm -rf ./dump
   ln -s ${Data_path}/preprocess_data/fastspeech2/dump/ ./
   echo -e "\033[31m data preprocess of fastspeech2_baker failed! \033[0m"
fi

sed -i "s/python3/python/g" ./local/train.sh
# rm -rf exp
export CUDA_VISIBLE_DEVICES=0
# ./local/train.sh ${conf_path} ${train_output_path} > train_1card.log 2>&1 &

if [ ! -f "pwg_baker_ckpt_0.4.zip" ]; then
   wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip
   unzip pwg_baker_ckpt_0.4.zip
fi
ckpt_name=snapshot_iter_30600.pdz
sed -i "s#python3#python#g" ./local/synthesize.sh
sed -i "s#python3#python#g" ./local/synthesize_e2e.sh
./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
