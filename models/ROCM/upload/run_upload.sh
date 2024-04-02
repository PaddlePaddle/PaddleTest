if [[ $1 =~ 'develop' ]]; then
    tar="develop"
elif [[ $1 =~ '.whl' ]]; then
    tar="publish"
else
    tar="release"
fi
python -m pip install pycrypto -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install bce-python-sdk -i https://pypi.tuna.tsinghua.edu.cn/simple
python upload.py /workspace/report_${tar}.tar  rocm/${tar}
# wget https://paddle-qa.bj.bcebos.com/rocm/release/report_release.tar
