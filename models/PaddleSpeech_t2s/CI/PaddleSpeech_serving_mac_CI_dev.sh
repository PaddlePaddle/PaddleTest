# start
basepath=`pwd`
mkdir log
log_path=${basepath}/log

printFun(){
    if [ $? -eq 0 ];then
    echo -e "\033[33m $1  predict  successfully!\033[0m"|tee -a $log_path/result.log
else
    # cat $log_path/serving.log
    echo -e "\033[31m $1 of predict failed!\033[0m"|tee -a $log_path/result.log
fi
}

killFun(){
ps aux | grep paddlespeech_server | awk '{print $2}' | xargs kill -9
}

displayFun(){
num=`cat $1 | grep -i "error" | wc -l`
if [ "${num}" -gt "0" ];then
cat $1
# echo -e "\033[31m $2  start failed!\033[0m"|tee -a $log_path/result.log
fi
}

# paddlespeech
python -m pip uninstall -y paddlespeech

python -m pip install .

unset http_proxy
unset https_proxy

cd demos/speech_server

if [ ! -f "zh.wav" ]; then
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
fi
# sed -i "s/device: /device: 'cpu'/g"  ./conf/application.yaml
paddlespeech_server start --config_file ./conf/application.yaml >> $log_path/offline_server.log 2>&1 &

sleep 120
cat $log_path/offline_server.log
echo '!!!'
ps aux | grep paddlespeech_server | grep -v grep
ps aux | grep paddlespeech_server | grep -v grep | wc -l
echo '!!!'
# asr
paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
printFun asr_offline
paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav

printFun tts_offline
paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
printFun cls_offline

# speaker vertification
if [ ! -f "85236145389.wav" ]; then
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/123456789.wav
fi

paddlespeech_client vector --task spk  --server_ip 127.0.0.1 --port 8090 --input 85236145389.wav
printFun vector_spk_offline
paddlespeech_client vector --task score  --server_ip 127.0.0.1 --port 8090 --enroll 85236145389.wav --test 123456789.wav
printFun vector_score_offline

# text
paddlespeech_client text --server_ip 127.0.0.1 --port 8090 --input "我认为跑步最重要的就是给我带来了身体健康"
printFun text_offline
displayFun $log_path/offline_server.log offline_server
killFun

## online_tts
cd ../streaming_tts_server
# http
paddlespeech_server start --config_file ./conf/tts_online_application.yaml >> $log_path/tts_online_http_server.log 2>&1 &
sleep 60
cat $log_path/tts_online_http_server.log

paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol http --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
printFun tts_online_http
displayFun $log_path/tts_online_http_server.log tts_online_http_server
killFun

# websocket
sed -i "" 's/http/websocket/g' ./conf/tts_online_application.yaml
# sed -i "s/device: 'cpu'/device: 'gpu:5'/g" ./conf/tts_online_application.yaml

paddlespeech_server start --config_file ./conf/tts_online_application.yaml >> $log_path/tts_online_websocket_server.log 2>&1 &
sleep 60
cat $log_path/tts_online_websocket_server.log
paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol websocket --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
printFun tts_online_websockert
displayFun $log_path/tts_online_websocket_server.log tts_online_websocket_server
killFun


### online_asr
cd ../streaming_asr_server
if [ ! -f "zh.wav" ]; then
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
fi

# sed -i "s/device: 'cpu' /device: 'gpu:5'/g"  ./conf/ws_conformer_wenetspeech_application.yaml
paddlespeech_server start --config_file ./conf/ws_conformer_wenetspeech_application.yaml >> $log_path/asr_online_websocket_server.log 2>&1 &
sleep 120

cat $log_path/asr_online_websocket_server.log
# asr
paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
printFun asr_online_websockert
displayFun $log_path/asr_online_websocket_server.log asr_online_websocket_server
killFun

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
