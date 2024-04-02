
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_generation/vae-seq2seq/
#临时环境更改
cd $root_path/models_repo && ls
cd $code_path
#获取数据逻辑
# wget https://paddlenlp.bj.bcebos.com/models/vae-seq2seq/imikolov_simple-examples.tgz
#python download.py --task ptb
#数据处理逻辑
