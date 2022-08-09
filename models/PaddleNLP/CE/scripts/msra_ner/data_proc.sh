
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

# 跑之前先修改下包的代码：
echo "删除之前先打印出来"
cat /usr/local/lib/python3.8/dist-packages/datasets/builder.py
sed -i '584,588d' /usr/local/lib/python3.8/dist-packages/datasets/builder.py
echo "删除之后再打印出来"
cat /usr/local/lib/python3.8/dist-packages/datasets/builder.py
