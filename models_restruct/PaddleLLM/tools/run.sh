export https_proxy=${proxy}
export http_proxy=${proxy}
export root_path=$PWD


####    测试框架下载    #####
wget -q ${CE_Link} --no-proxy 
unzip -q -P ${CE_pass} TestFrameWork.zip
####    测试case脚本下载    #####
# git clone https://github.com/Liujie0926/PaddleTest.git -b fix_CE
wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy 
tar xf PaddleTest.tar.gz
cp -r ./PaddleTest/models_restruct/PaddleLLM/. ./TestFrameWork/
rm -rf PaddleTest
####    套件库下载    #####
wget -q --no-proxy  https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddleNLP.tar --no-check-certificate
rm -rf PaddleNLP && tar xf PaddleNLP.tar && rm -rf PaddleNLP.tar 
# cd PaddleNLP && git fetch origin pull/10596/head:PR_10596 && git checkout PR_10596 && cd -
# fix CUDA error 801 with PR 10570 in docker 
sed -i '/from transformer_engine import transformer_engine_paddle as tex/,/^    }/d' PaddleNLP/paddlenlp/quantization/qat_utils.py
mv -v PaddleNLP ./TestFrameWork/PaddleLLM
unset http_proxy && unset https_proxy

# python -m pip install -r TestFrameWork/requirements.txt 
# python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda12.8-Cudnn9.7-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/3b5fe1f4e5b4bd71f1c0b8e33d459f2f4caff554/paddlepaddle_gpu-3.0.0.dev20250423-cp310-cp310-linux_x86_64.whl --force-reinstall --no-dependencies
####    for cuda12.8 pdc image    #####
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib/:${LD_LIBRARY_PATH}

####    case执行    #####
cd TestFrameWork
export models_file='tools/PaddleLLM_list'
export step="train:all+eval:all+infer:all+export:all+predict:all"
export reponame=PaddleLLM
export paddle_whl=None  #${paddle_whl}
export timeout=72000

####    for test    #####
# sed -i "s/1200/5/g" train/grpo.sh
# sed -i "s/1200/5/g" train/reinforce_plus_plus.sh
cp -r train/grpo.sh train/reinforce_plus_plus.sh PaddleLLM/llm/alignment/rl/
cp -r train/predict.sh train/infer.sh PaddleLLM/llm/
set +e
env | grep -i proxy

if ! pgrep -f reward_server.py > /dev/null; then
    echo "reward服务未运行"
    cd PaddleLLM/llm/alignment/rl/reward
    nohup python reward_server.py > reward_server.log 2>&1 &
    reward_pid=$!
    sleep 60s
    cd -
else
    echo "reward服务已运行"
fi

#paddlenlp_ops安装判断
whl_file=$(find "$root_path/../" -maxdepth 1 -type f -name "p*.whl" | head -n 1)
if [ -n "$whl_file" ]; then
    echo "发现文件：$whl_file，移动到PaddleLLM"
    mv "$whl_file" PaddleLLM/
else
    echo "未找到符合条件的 .whl 文件"
fi

# grpo
python main.py --models_file='tools/PaddleLLM_grpo' --step="${step:-train}" --reponame="${reponame:-PaddleClas}" --paddle_whl="${paddle_whl:-None}" --set_cuda='0,1,2,3' --timeout="${timeout:-3600}"  --plot='True' > run_grpo.log 2>&1 &
grpo_pid=$!
sleep 100s
# rf++
python main.py --models_file='tools/PaddleLLM_rf++' --step="${step:-train}" --reponame="${reponame:-PaddleClas}" --paddle_whl="${paddle_whl:-None}" --set_cuda='4,5,6,7' --timeout="${timeout:-3600}"  --plot='True' > run_rf++.log 2>&1 &
rf_pid=$!

wait $grpo_pid
wait $rf_pid
echo "kill reward 服务"
kill -9 $reward_pid || pkill -9 -f reward_server.py

#wget https://xly-devops.bj.bcebos.com/tools/allure-2.19.0.zip && unzip allure-2.19.0.zip 
cp -r /root/paddlejob/workspace/env_run/agent/allure-2.19.0 ./
allure-2.19.0/bin/allure generate result/ -o report

bash tools/generate_template_llm.sh
# cd utils && python emails.py --reponame PaddleLLM --reporturl_name default_template_llm  --email_addr liujie44@baidu.com --email_sub PaddleLLM_CE && cd -
cd utils && python emails.py --reponame PaddleLLM --reporturl_name default_template_llm  --email_addr liujie44@baidu.com,fangzeyang@baidu.com,gongenlei@baidu.com,liujihua@baidu.com,yujun06@baidu.com,zhonghui03@baidu.com,zhuweiguo@baidu.com --email_sub PaddleLLM_CE && cd -

cp -r $root_path /root/paddlejob/workspace/env_run/agent/workspace/history/