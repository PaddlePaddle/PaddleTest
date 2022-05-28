@ echo off
set log_path=log

echo "*****************create_virtual_env*********"
conda create -n paddlespeech_env python=3.8 -y
conda activate paddlespeech_env
echo "*****************python_version****"
python -c "import sys; print('python version:',sys.version_info[:])";

set no_proxy=bcebos.com
set http_proxy=%proxy%
set https_proxy=%proxy%
@echo on
echo "*****************speech_version****"
git rev-parse HEAD
conda install -y -c conda-forge sox libsndfile bzip2
python -m pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install $1 --ignore-installed
python -m pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
echo  "*****************paddle_version*****"
python -c "import paddle; print(paddle.__version__,paddle.version.commit)"
cd tests/unit/cli
chdir

echo "########test_cli_cls########"
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav --no-check-certificate
paddlespeech cls --input ./cat.wav --topk 10 > ..\..\..\%log_path%\cli_cls.log 2>&1
if not !errorlevel! == 0 (
        echo  cli,cls,FAIL  >> ..\..\..\%log_path%\result.log
        echo  cli_cls failed!
) else (
        echo  cli,cls,SUCCESS  >> ..\..\..\%log_path%\result.log
        echo  cli_cls successfully!
)
echo "########test_cli_text########"
paddlespeech text --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭 > ..\..\..\%log_path%\cli_text.log 2>&1
if not !errorlevel! == 0 (
        echo  cli,text,FAIL  >> ..\..\..\%log_path%\result.log
        echo  cli_text failed!
) else (
        echo  cli,text,SUCCESS  >> ..\..\..\%log_path%\result.log
        echo  cli_text successfully!
)
echo "########test_cli_asr########"
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav --no-check-certificate
paddlespeech asr --input ./zh.wav > ..\..\..\%log_path%\cli_asr.log 2>&1
if not !errorlevel! == 0 (
        echo  cli,asr,FAIL  >> ..\..\..\%log_path%\result.log
        echo  cli_asr failed!
) else (
        echo  cli,asr,SUCCESS  >> ..\..\..\%log_path%\result.log
        echo  cli_asr successfully!
)
echo "########test_cli_tts########"
paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" > ..\..\..\%log_path%\cli_tts.log 2>&1
if not !errorlevel! == 0 (
        echo  cli,tts,FAIL  >> ..\..\..\%log_path%\result.log
        echo  cli_tts failed!
) else (
        echo  cli,tts,SUCCESS  >> ..\..\..\%log_path%\result.log
        echo  cli_tts successfully!
)

echo "########exit_environment########"
conda deactivate
echo "envir before"
conda info --envs
conda remove -n paddlespeech_env --all -y
echo "envir after"
conda info --envs

cd ../../../
chdir
rem exit
set num=0
for /F %%i in ('findstr /s "FAIL" %log_path%/result.log') do ( set num=%%i )
findstr /s "FAIL" %log_path%/result.log
if %num%==0 (exit /b 0)else (exit /b 1)
