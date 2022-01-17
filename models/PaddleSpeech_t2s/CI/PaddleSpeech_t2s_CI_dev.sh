unset GREP_OPTIONS
# echo ${cudaid1}
echo ${cudaid2}
echo ${Data_path}
echo ${paddle_compile}

echo "######  ---python  env -----"
rm -rf /usr/local/python2.7.15/bin/python
rm -rf /usr/local/bin/python
export PATH=/usr/local/bin/python:${PATH}
case $1 in #python
36)
ln -s /usr/local/bin/python3.6 /usr/local/bin/python
;;
37)
ln -s /usr/local/bin/python3.7 /usr/local/bin/python
;;
38)
ln -s /usr/local/bin/python3.8 /usr/local/bin/python
;;
39)
ln -s /usr/local/bin/python3.9 /usr/local/bin/python
;;
esac
python -c "import sys; print('python version:',sys.version_info[:])";

echo "######  ----install  paddle-----"
unset http_proxy
unset https_proxy
python -m pip uninstall paddlepaddle-gpu -y
python -m pip install $4 #paddle_compile
num=`python -m pip list | grep paddlepaddle | wc -l`
if [ "${num}" -eq "0" ]; then
   wget https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxCentos_Gcc82_Cuda10.2_Trton_Py37_Compile_D_Develop/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
   python -m pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
fi
echo "######  ----paddle version-----"
python -c "import paddle; print(paddle.version.commit)";

#system
if [ -d "/etc/redhat-release" ]; then
   echo "######  system centos"
else
   echo "######  system linux"
fi

# env
#export FLAGS_fraction_of_gpu_memory_to_use=0.8
# dependency
unset http_proxy
unset https_proxy
python -m pip install --ignore-installed --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade --force --ignore-installed . -i https://pypi.tuna.tsinghua.edu.cn/simple

# dir
log_path=log
rm -rf log/
stage_list='train synthesize synthesize_e2e inference'
for stage in  ${stage_list}
do
if [ -d ${log_path}/${stage} ]; then
   echo -e "\033[33m ${log_path}/${stage} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${stage}
   echo -e "\033[33m ${log_path}/${stage} is created successfully!\033[0m"
fi
done
echo "`python -m pip list | grep paddle`" |tee -a ${log_path}/result.log
python -c 'import paddle;print(paddle.version.commit)' |tee -a ${log_path}/result.log

echo -e "fastspeech2\nparallelwavegan\nspeedyspeech\ntacotron2\ntransformertts\nwaveflow" > models_list_all
shuf models_list_all > models_list_shuf
head -n 2 models_list_shuf > models_list

cat models_list | while read line
do
echo $line
case $line in

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

tacotron2)
# tacotron2
cd examples/ljspeech/tts0

source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
gpus=$2

rm -rf ./preprocessed_ljspeech
ln -s $3/preprocess_data/tacotron2/preprocessed_ljspeech/ ./
train_output_path=output
rm -rf ${train_output_path}
export CUDA_VISIBLE_DEVICES=${gpus}
python ${BIN_DIR}/train.py --data=preprocessed_ljspeech --output=${train_output_path} --ngpu=2 --opts data.batch_size 2 training.max_iteration 10 training.valid_interval 10 training.save_interval 10 > ../../../$log_path/train/tacotron2.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m training of tacotron2 successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/train/tacotron2.log
   echo -e "\033[31m training of tacotron2 failed! \033[0m" | tee -a ../../../$log_path/result.log
fi

rm -rf exp
head -3 ${BIN_DIR}/../sentences_en.txt > sentences_en3.txt
sed -i 's#python3#python#g;s#${BIN_DIR}/../sentences_en.txt#./sentences_en3.txt#g' ./local/synthesize.sh
ckpt_name=step-10
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${train_output_path} ${ckpt_name} > ../../../$log_path/synthesize/tacotron2.log 2>&1
if [ $? -eq 0 ];then
   echo -e "\033[33m synthesize of tacotron2 successfully! \033[0m" | tee -a ../../../$log_path/result.log
else
   cat ../../../$log_path/synthesize/tacotron2.log
   echo -e "\033[31m synthesize of tacotron2 failed! \033[0m" | tee -a ../../../$log_path/result.log
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

input_mel_path=../tts0/output/test
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
