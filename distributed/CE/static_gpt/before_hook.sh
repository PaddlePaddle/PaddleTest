echo "Run before_hook.sh ..."
unlink /usr/bin/python3
ln -sf /usr/bin/python3.7 /usr/bin/python3
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export PADDLE_WITH_GLOO=0

echo "install dependent package"
python3 -m pip install jieba -i https://mirror.baidu.com/pypi/simple
python3 -m pip install Pillow
python3 -m pip install h5py
python3 -m pip install seqeval sentencepiece
python3 -m pip install colorlog colorama
python3 -m pip install regex
python3 -m pip install multiprocess -i https://mirror.baidu.com/pypi/simple
python3 -m pip install tqdm
python3 -m pip install visualdl

echo "uninstall paddlepaddle"
python3 -m pip uninstall -y paddlepaddle-gpu
python3 -m pip uninstall -y paddlepaddle

echo "install paddlepaddle"
python3 -m pip uninstall paddlepaddle-gpu -y
#hadoop fs -D fs.default.name=afs://yinglong.afs.baidu.com:9902 -D hadoop.job.ugi=paddle,dltp_paddle@123 -get /user/paddle/liujie44/auto_parallel/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl .
wget https://paddle-wheel.bj.bcebos.com/develop/linux/gpu-cuda11.0-cudnn8-mkl_gcc8.2/paddlepaddle_gpu-0.0.0.post110-cp37-cp37m-linux_x86_64.whl --no-check-certificate
export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
python3 -m pip install paddlepaddle_gpu-0.0.0.post110-cp37-cp37m-linux_x86_64.whl
# debug or release
#wget http://gzns-ps-201608-m02-www028.gzns.baidu.com:8987/images/2.1.2/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl --no-check-certificate
#python3 -m pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
echo "install paddle ${new_whl_name} success.."

hadoop fs -D fs.default.name=afs://yinglong.afs.baidu.com:9902 -D hadoop.job.ugi=paddle,dltp_paddle@123 -get /user/paddle/liujie44/static_gpt.tar.gz .
tar zxf static_gpt.tar.gz
cd static_gpt/manual_auto_parallel_model/gpt/
mkdir data && cd data
$hadoop fs -D fs.default.name=afs://yinglong.afs.baidu.com:9902 -D hadoop.job.ugi=paddle,dltp_paddle@123 -get /user/paddle/liujie44/train.data.json_ids.npz .
cd ..
echo $PWD
ls -a

echo "End before_hook.sh ..."
