python -m pip install --no-cache-dir --upgrade pip
set -x
repo_list='PaddleClas PaddleOCR PaddleDetection PaddleSeg PaddleNLP PaddleSpeech PaddleHub'
for repo in $repo_list
do
echo $repo
git clone http://github.com/PaddlePaddle/$repo.git
cd  $repo

if [ "$repo" != "PaddleSpeech" ];then
python -m pip install -r requirements.txt
fi

if  [ "$repo" = "PaddleOCR" ];then
python -m pip install -r ppstructure/recovery/requirements.txt
fi

python -m pip install .
cd ..
done
