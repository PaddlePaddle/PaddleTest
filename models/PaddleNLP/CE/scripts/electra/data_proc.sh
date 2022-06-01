
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"


#配置目标数据存储路径
code_path=${nlp_dir}/model_zoo/$model_name

#获取数据逻辑
rm -rf $code_path/BookCorpus/
mkdir -p $code_path/BookCorpus/
#here need bos url
wget -P $code_path/BookCorpus https://paddlenlp.bj.bcebos.com/models/electra/train.data.txt
#cp /home/xishengdai/train.data $code_path/BookCorpus/

cd $code_path/BookCorpus
mv train.data.txt train.data
