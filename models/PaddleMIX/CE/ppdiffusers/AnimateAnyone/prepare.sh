# 安装其他所需的依赖, 若提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt

# 下载预训练模型
python scripts/download_weights.py

# ubc_fashion数据集下载
wget https://bj.bcebos.com/paddlenlp/models/community/tsaiyue/ubcNbili_data/ubcNbili_data.tar.gz

# 文件解压
tar -xzvf ubcNbili_data.tar.gz
