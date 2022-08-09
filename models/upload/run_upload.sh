python -m pip install pycrypto -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install bce-python-sdk -i https://pypi.tuna.tsinghua.edu.cn/simple
python upload.py /workspace/report_$1.tar  rocm/$1
# wget https://paddle-qa.bj.bcebos.com/rocm/release/report_release.tar
