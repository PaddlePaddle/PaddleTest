echo  '*****************paddle_version*****'
python -c "import paddle; print(paddle.__version__,paddle.version.commit)";
echo '*****************clas_version****'
git rev-parse HEAD

echo "Data_path Project_path"
echo ${Data_path}
ls ${Data_path}
echo ${Project_path}
ls ${Project_path} |head -n 2

echo "path before"
pwd
cd ${Project_path}
echo "path after"
pwd

unset http_proxy
unset https_proxy
python -m pip install --upgrade --force --ignore-installed  paddleslim  -i https://mirror.baidu.com/pypi/simple
python -m pip install --upgrade --force --ignore-installed  -r requirements.txt  -i https://mirror.baidu.com/pypi/simple
python -m pip list


#需要商榷  ls先找到hadoop的路径，再进行软链
# cd dataset
# ls
# wget http://10.89.242.17:8903/ssd1/gry_data/ILSVRC2012_w.tar >download_ILSVRC2012.log 2>&1
# tar xf ILSVRC2012_w.tar
# mv ILSVRC2012_w  ILSVRC2012
# ls
# ls ILSVRC2012
# cd ..

#先用一个flower数据试一下
cd dataset
cp ${Data_path}/flowers102.zip .
if [[ -f "flowers102.zip" ]];then
   unzip flowers102.zip >unzip_flowers102.log 2>&1
   ls
else
   wget https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip >download_flowers102.log 2>&1
   unzip flowers102.zip >unzip_flowers102.log 2>&1
   ls
fi
cd ..

log_path=log
phases='train'
for phase in $phases
do
if [[ -d ${log_path}/${phase} ]]; then
   echo -e "\033[33m ${log_path}/${phase} is exsit!\033[0m"
else
   mkdir -p  ${log_path}/${phase}
   echo -e "\033[33m ${log_path}/${phase} is created successfully!\033[0m"
fi
done

filename=${1##*/}
#echo $filename
model=${filename%.*}
echo $model

echo "nvidia"
sleep 10
echo $CUDA_VISIBLE_DEVICES
unset PYTHONPATH

nvidia-smi

# python -m paddle.distributed.launch tools/train.py -c $1  > $model.log

python -m paddle.distributed.launch tools/train.py -c $1 \
    -o DataLoader.Train.dataset.cls_label_path="./dataset/flowers102/train_list.txt" \
    -o DataLoader.Train.dataset.image_root="./dataset/flowers102/" \
    -o DataLoader.Eval.dataset.cls_label_path="./dataset/flowers102/val_list.txt" \
    -o DataLoader.Eval.dataset.image_root="./dataset/flowers102/" \
    -o Arch.pretrained=True -o DataLoader.Train.sampler.batch_size=32 \
    -o DataLoader.Eval.sampler.batch_size=32 > $log_path/train/${model}_convergence.log 2>&1


sleep 10

nvidia-smi

cat $log_path/train/${model}_convergence.log
cat $log_path/train/${model}_convergence.log |grep Avg
