
#获取当前路径
cur_path=`pwd`
echo "开始编译PaddleNLP包"

#配置目标数据存储路径
root_path=$cur_path/../
modle_path=$cur_path/../models_repo/
# 编包
cd $modle_path
python setup.py bdist_wheel
python -m pip install --upgrade paddlenlp
python -m pip uninstall -y paddlenlp
python -m pip install --ignore-installed ./dist/paddlenlp*.whl
# 如果前面编包失败就
if [ $? -ne 0 ];then
    python -m pip install ./
fi
