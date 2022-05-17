echo  '*****************paddle_version*****'
python -c "import paddle; print(paddle.__version__,paddle.version.commit)";
echo '*****************clas_version****'
git rev-parse HEAD

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
ls
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip >download_flowers102.log 2>&1
unzip flowers102.zip >unzip_flowers102.log 2>&1
ls
cd ..


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
    -o DataLoader.Eval.sampler.batch_size=32 > $model.log


sleep 10

nvidia-smi

cat $model.log
cat $model.log |grep Avg
