unset GREP_OPTIONS
# echo ${cudaid1}
echo ${cudaid2}
echo ${Data_path}
echo ${paddle_compile}
echo ${model_flag}

mkdir run_env_py37;
ln -s $(which python3.7) run_env_py37/python;
ln -s $(which pip3.7) run_env_py37/pip;
export PATH=$(pwd)/run_env_py37:${PATH};
export http_proxy=${http_proxy}
export https_proxy=${https_proxy}
export no_proxy=bcebos.com;
python -m pip install pip==20.2.4 --ignore-installed;
python -m pip install $4 --no-cache-dir --ignore-installed;
apt-get update
if [[ $5 == 'all' ]];then
   apt-get install -y sox pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev
fi
pushd tools; make virtualenv.done; popd
if [ $? -ne 0 ];then
    exit 1
fi
source tools/venv/bin/activate
python -m pip install pip==20.2.4 --ignore-installed;
python -m pip install $4 --no-cache-dir
python -m pip install numpy==1.20.1 --ignore-installed
python -m pip install pyparsing==2.4.7 --ignore-installed
#pip install -e .
pip install .
python -c "import sys; print('python version:',sys.version_info[:])";

#system
if [ -d "/etc/redhat-release" ]; then
   echo "######  system centos"
else
   echo "######  system linux"
fi

# dir
log_path=log
stage_list='train synthesize synthesize_e2e inference'
for stage in  ${stage_list}
do
if [ -d ${log_path}/${stage} ]; then
   echo -e "\033[33m ${log_path}/${stage} is exist!\033[0m"
else
   mkdir -p  ${log_path}/${stage}
   echo -e "\033[33m ${log_path}/${stage} is created successfully!\033[0m"
fi
done
echo "`python -m pip list | grep paddle`" |tee -a ${log_path}/result.log
python -c 'import paddle;print(paddle.version.commit)' |tee -a ${log_path}/result.log

echo -e "newTacotron2\nspeedyspeech\nfastspeech2\nparallelwavegan\nStyleMelGAN\nHiFiGAN\nWaveRNN\ntransformertts\nwaveflow" > models_list_all
if [[ $5 == 'pr' ]];then
   echo "#### model_flag pr"
   shuf models_list_all > models_list_shuf
   head -n 2 models_list_shuf > models_list
else
   echo "#### model_flag all"
   shuf models_list_all > models_list
fi

cat models_list | while read line
do
echo $line
case $line in

newTacotron2)
# tacotron2_csmsc
cd examples/csmsc/tts0
rm -rf ./dump
ln -s $3/preprocess_data/new_tacotron2/dump/ ./

source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2
conf_path=conf/default.yaml
train_output_path=exp/default
sed -i "s/max_epoch: 200/max_epoch: 1/g" ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} > ../../../$log_path/train/tacotron2_csmsc.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of tacotron2_csmsc successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/tacotron2_csmsc.log
   echo -e "\033[31m training of tacotron2_csmsc failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

if [ ! -f "pwg_baker_ckpt_0.4.zip" ]; then
   wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip
   unzip pwg_baker_ckpt_0.4.zip
fi
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
ckpt_name=snapshot_iter_76.pdz
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_10.jsonl#g;s#python3#python#g" ./local/synthesize.sh
# synthesize, vocoder is pwgan
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/tacotron2_csmsc.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of tacotron2_csmsc successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/tacotron2_csmsc.log
   echo -e "\033[31m synthesize of tacotron2_csmsc failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

head -5 ${BIN_DIR}/../sentences.txt > sentences_5.txt
sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences.txt#./sentences_5.txt#g' ./local/synthesize_e2e.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize_e2e/tacotron2_csmsc.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize_e2e of tacotron2_csmsc successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize_e2e/tacotron2_csmsc.log
   echo -e "\033[31m synthesize_e2e of tacotron2_csmsc failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences.txt#./sentences_5.txt#g' ./local/inference.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path} > ../../../$log_path/inference/tacotron2_csmsc.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m inference of tacotron2_csmsc successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/inference/tacotron2_csmsc.log
   echo -e "\033[31m inference of tacotron2_csmsc failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


speedyspeech)
# speedyspeech_csmsc
cd examples/csmsc/tts2

source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
ln -s $3/preprocess_data/speedyspeech/dump/ ./
sed -i "s/max_epoch: 200/max_epoch: 1/g" ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}  > ../../../$log_path/train/speedyspeech.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of speedyspeech_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/speedyspeech.log
   echo -e "\033[31m training of speedyspeech_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

if [ ! -f "pwg_baker_ckpt_0.4.zip" ]; then
   wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip
   unzip pwg_baker_ckpt_0.4.zip
fi
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
ckpt_name=snapshot_iter_76.pdz
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_10.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/speedyspeech.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of speedyspeech_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/speedyspeech.log
   echo -e "\033[31m synthesize of speedyspeech_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

head -5 ${BIN_DIR}/../sentences.txt > sentences_5.txt
sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences.txt#./sentences_5.txt#g' ./local/synthesize_e2e.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize_e2e/speedyspeech.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize_e2e of speedyspeech_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize_e2e/speedyspeech.log
   echo -e "\033[31m synthesize_e2e of speedyspeech_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences.txt#./sentences_5.txt#g' ./local/inference.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path} > ../../../$log_path/inference/speedyspeech.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m inference of speedyspeech_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/inference/speedyspeech.log
   echo -e "\033[31m inference of speedyspeech_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


fastspeech2)
# fastspeech2_csmsc
cd examples/csmsc/tts3
mkdir ~/datasets
ln -s $3/train_data/BZNSYP/ ~/datasets/
rm -rf ./dump
ln -s $3/preprocess_data/fastspeech2/dump/ ./

source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
sed -i "s/max_epoch: 1000/max_epoch: 1/g" ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}  > ../../../$log_path/train/fastspeech2.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of fastspeech2_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/fastspeech2.log
   echo -e "\033[31m training of fastspeech2_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

if [ ! -f "pwg_baker_ckpt_0.4.zip" ]; then
   wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip
   unzip pwg_baker_ckpt_0.4.zip
fi
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
ckpt_name=snapshot_iter_76.pdz
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_10.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/fastspeech2.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of fastspeech2_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/fastspeech2.log
   echo -e "\033[31m synthesize of fastspeech2_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

head -5 ${BIN_DIR}/../sentences.txt > sentences_5.txt
sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences.txt#./sentences_5.txt#g' ./local/synthesize_e2e.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize_e2e/fastspeech2.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize_e2e of fastspeech2_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize_e2e/fastspeech2.log
   echo -e "\033[31m synthesize_e2e of fastspeech2_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences.txt#./sentences_5.txt#g' ./local/inference.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path} > ../../../$log_path/inference/fastspeech2.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m inference of fastspeech2_baker successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/inference/fastspeech2.log
   echo -e "\033[31m inference of fastspeech2_baker failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


parallelwavegan)
# parallel_wavegan_csmsc
cd examples/csmsc/voc1
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
ln -s $3/preprocess_data/pwg/dump/ ./

sed -i "s/train_max_steps: 400000/train_max_steps: 10/g;s/save_interval_steps: 5000/save_interval_steps: 10/g;s/eval_interval_steps: 1000/eval_interval_steps: 10/g"  ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} > ../../../$log_path/train/pwg.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of parallel_wavegan successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/pwg.log
   echo -e "\033[31m training of parallel_wavegan failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

ckpt_name=snapshot_iter_10.pdz
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_10.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/pwg.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of parallel_wavegan successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/pwg.log
   echo -e "\033[31m synthesize of parallel_wavegan failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


MultiBandMelGAN)
# MultiBand MelGAN csmsc
cd examples/csmsc/voc3
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
ln -s $3/preprocess_data/pwg/dump/ ./

sed -i "s/train_max_steps: 1000000/train_max_steps: 10/g;s/save_interval_steps: 5000/save_interval_steps: 10/g;s/eval_interval_steps: 1000/eval_interval_steps: 10/g"  ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} > ../../../$log_path/train/MultiBand_MelGAN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of MultiBand_MelGAN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/MultiBand_MelGAN.log
   echo -e "\033[31m training of MultiBand_MelGAN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

ckpt_name=snapshot_iter_10.pdz
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_10.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/MultiBand_MelGAN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of MultiBand_MelGAN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/MultiBand_MelGAN.log
   echo -e "\033[31m synthesize of MultiBand_MelGAN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


StyleMelGAN)
# Style MelGAN csmsc
cd examples/csmsc/voc4
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
ln -s $3/preprocess_data/pwg/dump/ ./

sed -i "s/train_max_steps: 1500000/train_max_steps: 10/g;s/save_interval_steps: 5000/save_interval_steps: 10/g;s/eval_interval_steps: 1000/eval_interval_steps: 10/g;s/batch_size: 32/batch_size: 16/g"  ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} > ../../../$log_path/train/Style_MelGAN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of Style_MelGAN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/Style_MelGAN.log
   echo -e "\033[31m training of Style_MelGAN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

ckpt_name=snapshot_iter_10.pdz
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_10.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/Style_MelGAN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of Style_MelGAN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/Style_MelGAN.log
   echo -e "\033[31m synthesize of Style_MelGAN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


HiFiGAN)
# HiFiGAN csmsc
cd examples/csmsc/voc5
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
ln -s $3/preprocess_data/pwg/dump/ ./

sed -i "s/train_max_steps: 2500000/train_max_steps: 10/g;s/save_interval_steps: 5000/save_interval_steps: 10/g;s/eval_interval_steps: 1000/eval_interval_steps: 10/g"  ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} > ../../../$log_path/train/HiFiGAN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of HiFiGAN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/HiFiGAN.log
   echo -e "\033[31m training of HiFiGAN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

ckpt_name=snapshot_iter_10.pdz
head -10 ./dump/test/norm/metadata.jsonl > ./metadata_10.jsonl
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_10.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/HiFiGAN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of HiFiGAN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/HiFiGAN.log
   echo -e "\033[31m synthesize of HiFiGAN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


WaveRNN)
# WaveRNN csmsc
cd examples/csmsc/voc6
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
ln -s $3/preprocess_data/pwg/dump/ ./

sed -i "s/train_max_steps: 400000/train_max_steps: 10/g;s/save_interval_steps: 5000/save_interval_steps: 10/g;s/eval_interval_steps: 1000/eval_interval_steps: 10/g;s/batch_size: 64/batch_size: 32/g"  ./conf/default.yaml
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
rm -rf ./exp
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} > ../../../$log_path/train/WaveRNN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of WaveRNN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/WaveRNN.log
   echo -e "\033[31m training of WaveRNN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

ckpt_name=snapshot_iter_10.pdz
head -3 ./dump/test/norm/metadata.jsonl > ./metadata_3.jsonl
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_3.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/WaveRNN.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of WaveRNN successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/WaveRNN.log
   echo -e "\033[31m synthesize of WaveRNN failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


transformertts)
# transformer tts
cd examples/ljspeech/tts1

source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

conf_path=conf/default.yaml
train_output_path=exp/default
rm -rf ./dump
ln -s $3/preprocess_data/transformer_tts/dump ./dump
rm -rf ./exp
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
sed -i "s/max_epoch: 500/max_epoch: 1/g;s/batch_size: 16/batch_size: 4/g"  ./conf/default.yaml
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} > ../../../$log_path/train/transformer_tts.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of transformer tts successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/transformer_tts.log
   echo -e "\033[31m training of transformer tts failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

if [ ! -f "waveflow_ljspeech_ckpt_0.3.zip" ]; then
   wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/waveflow/waveflow_ljspeech_ckpt_0.3.zip
   unzip waveflow_ljspeech_ckpt_0.3.zip
fi
ckpt_name=snapshot_iter_1612.pdz
head -3 ./dump/test/norm/metadata.jsonl > ./metadata_3.jsonl
sed -i "s#dump/test/norm/metadata.jsonl#./metadata_3.jsonl#g;s#python3#python#g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/transformer_tts.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of transformer tts successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/transformer_tts.log
   echo -e "\033[31m synthesize of transformer tts failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

head -3 ${BIN_DIR}/../sentences_en.txt > sentences_en3.txt
sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences_en.txt#./sentences_en3.txt#g' ./local/synthesize_e2e.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize_e2e/transformer_tts.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize_e2e of transformer tts successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize_e2e/transformer_tts.log
   echo -e "\033[31m synthesize_e2e of transformer tts failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;


waveflow)
# waveflow
cd examples/ljspeech/voc0

source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

rm -rf ./preprocessed_ljspeech
ln -s $3/preprocess_data/waveflow/preprocessed_ljspeech/ ./
preprocess_path=preprocessed_ljspeech
train_output_path=output
rm -rf output
sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh
export CUDA_VISIBLE_DEVICES=${gpus}
python ${BIN_DIR}/train.py --data=${preprocess_path} --output=${train_output_path} --ngpu=2 --opts data.batch_size 2 training.max_iteration 10 training.valid_interval 10 training.save_interval 10 > ../../../$log_path/train/waveflow.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of waveflow successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/waveflow.log
   echo -e "\033[31m training of waveflow failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

ln -s $3/mel_test/ ./
input_mel_path=mel_test/
ckpt_name=step-10
sed -i "s/python3/python/g" ./local/synthesize.sh
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${input_mel_path} ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/waveflow.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of waveflow successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/waveflow.log
   echo -e "\033[31m synthesize of waveflow failed! \033[0m" | tee -a ../../../$log_path/result.log
fi
cd ../../..
;;

esac
done

if [[ $5 == 'all' ]];then
   # test_cli
   cd tests/unit/cli
   python -m pip uninstall importlib-metadata -y
   python -m pip install importlib-metadata==2.0.0
   export CUDA_VISIBLE_DEVICES=${gpus}
   bash test_cli.sh > ../../../$log_path/test_cli.log 2>&1
   if [ $? -eq 0 ];then
      echo -e "\033[33m test_cli successfully! \033[0m" | tee -a ../../../$log_path/result.log
   else
      cat ../../../$log_path/test_cli.log
      echo -e "\033[31m test_cli failed! \033[0m" | tee -a ../../../$log_path/result.log
   fi
   cd ../../..
fi

# result
num=`cat $log_path/result.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
echo -e "-----------------------------base cases-----------------------------"
cat $log_path/result.log | grep "failed"
echo -e "--------------------------------------------------------------------"
exit 1
else
exit 0
fi
