pddleSpeech/demos/speech_server
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
rem sed -i "s/device: /device: 'cpu'/g"  ./conf/application.yaml
start paddlespeech_server start --config_file ./conf/application.yaml 2>&1 &
timeout /nobreak /t 120

rem asr
paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input ./zh.wav

rem speaker vertification
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
rem cp 85236145389.wav 123456789.wav
paddlespeech_client vector --task spk  --server_ip 127.0.0.1 --port 8090 --input 85236145389.wav
paddlespeech_client vector --task score  --server_ip 127.0.0.1 --port 8090 --enroll 85236145389.wav --test 85236145389.wav

rem text
paddlespeech_client text --server_ip 127.0.0.1 --port 8090 --input "我认为跑步最重要的就是给我带来了身体健康"

taskkill /f /im paddlespeech_server*
rem ps aux | grep server | awk '{print $2}' | xargs kill -9
s aux | grep paddlespeech_server | awk '{print $2}' | xargs kill -9

