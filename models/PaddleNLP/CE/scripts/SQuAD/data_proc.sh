
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#获取数据&模型逻辑
# 删除之前先打印出来
echo "删除之前先打印出来"
cat /opt/_internal/cpython-3.7.0/lib/python3.7/site-packages/datasets/builder.py
sed -i '584,588d' /opt/_internal/cpython-3.7.0/lib/python3.7/site-packages/datasets/builder.py
echo "删除之后再打印出来"
cat /opt/_internal/cpython-3.7.0/lib/python3.7/site-packages/datasets/builder.py
