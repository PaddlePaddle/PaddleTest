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
# fix protobuf upgrade
python -m pip uninstall protobuf -y
python -m pip install protobuf==3.20.1
python -m pip list | grep protobuf
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
python -c 'import paddle;print(paddle.version.commit)'

# serving
python -m pip install .
cd demos/speech_server
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
# sed -i "s/device: /device: 'cpu'/g"  ./conf/application.yaml
paddlespeech_server start --config_file ./conf/application.yaml 2>&1 &

sleep 60
# asr
paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input ./zh.wav

# speaker vertification
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
cp 85236145389.wav 123456789.wav
paddlespeech_client vector --task spk  --server_ip 127.0.0.1 --port 8090 --input 85236145389.wav
paddlespeech_client vector --task score  --server_ip 127.0.0.1 --port 8090 --enroll 85236145389.wav --test 123456789.wav

# text
paddlespeech_client text --server_ip 127.0.0.1 --port 8090 --input "我认为跑步最重要的就是给我带来了身体健康"


ps aux | grep paddlespeech_server | awk '{print $2}' | xargs kill -9
