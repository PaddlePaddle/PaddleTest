# python
rm -rf /usr/local/bin/python;
ln -s /opt/conda/envs/python35-paddle120-env/bin/python3.7 /usr/local/bin/python;

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --upgrade parl opencv-python

set -x
repo_list='PaddleClas PaddleOCR PaddleDetection PaddleSeg PaddleNLP PaddleSpeech PaddleHub'
for repo in $repo_list
do
echo $repo
git clone http://github.com/PaddlePaddle/$repo.git
cd  $repo

if [ "$repo" != "PaddleSpeech" ];then
python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
fi

if  [ "$repo" = "PaddleOCR" ];then
python -m pip install -r ppstructure/recovery/requirements.txt -i https://mirror.baidu.com/pypi/simple
fi

python -m pip install .
cd ..
done
