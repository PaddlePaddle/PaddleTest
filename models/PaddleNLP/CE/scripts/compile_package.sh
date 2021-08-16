
#获取当前路径
cur_path=`pwd`
echo "开始编译PaddleNLP包"

#配置目标数据存储路径
root_path=$cur_path/../
modle_path=$cur_path/../models_repo/
# 编包
cd $modle_path
python setup.py bdist_wheel
python -m pip install ./dist/paddlenlp*.whl
