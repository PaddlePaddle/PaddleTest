unlink /usr/bin/python3
ln -sf /usr/bin/python3.7 /usr/bin/python3
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export PADDLE_WITH_GLOO=0
export DATA_DIR=./data
#export PYTHONPATH=$PYTHONPATH:../../../../hybrid
export NCCL_DEBUG=INFO

echo "current path"
pwd
cd static_gpt/manual_auto_parallel_model/gpt/

sh run_static.sh $1 $4
sh run_static.sh $2 $5
sh run_static.sh $3 $6
