
# ubuntu
if [ -f "/etc/lsb-release" ];then
apt-get update -y
apt-get -y install libsndfile1
# centos
elif [ -f "/etc/redhat-release" ];then
yum update -y
yum install -y libsndfile
fi

data_path=/paddle/data/ce_data/PaddleSpeech_t2s
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
wget https://paddlespeech.bj.bcebos.com/Parakeet/tools/nltk_data.tar.gz
tar xf nltk_data.tar.gz
rm -rf *.tar.gz
rm -rf /root/nltk_data
mv nltk_data /root

# python -m pip install .
mkdir log
# fastspeech2 2card
cd examples/csmsc/tts3
rm -rf ./dump
echo ${data_path}/preprocess_data/fastspeech2/dump
ln -s ${data_path}/preprocess_data/fastspeech2/dump/ ./
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=6,7
conf_path=conf/default.yaml
train_output_path=exp/default

sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}  > ../../../log/fastspeech2_2card.log 2>&1 &
cd ../../..

# speedyspeech 1card
cd examples/csmsc/tts2
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=5

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
echo ${data_path}/preprocess_data/speedyspeech/dump/
ln -s ${data_path}/preprocess_data/speedyspeech/dump/ ./
sed -i "s/python3/python/g;s/ngpu=2/ngpu=1/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}  > ../../../speedyspeech_1card.log 2>&1 &
cd ../../..
