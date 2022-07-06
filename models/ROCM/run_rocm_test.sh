# paddle
rm -rf paddlepaddle_rocm*.whl*
echo $1
if [[ $1 =~ 'develop' ]]; then
    wget -q https://paddle-wheel.bj.bcebos.com/develop/dcu1/paddlepaddle_rocm-0.0.0.dev401-cp37-cp37m-linux_x86_64.whl
    whl=`ls |grep ".whl"`
    echo $whl
    python -m pip uninstall -y paddlepaddle-rocm
    python -m pip install $whl -i https://mirror.baidu.com/pypi/simple
    tar="develop"
elif [[ $1 =~ '.whl' ]]; then
    wget -q $1
    whl=`ls |grep ".whl"`
    echo $whl
    python -m pip uninstall -y paddlepaddle-rocm
    python -m pip install $whl -i https://mirror.baidu.com/pypi/simple
    tar="publish"
else
    wget -q https://paddle-qa.bj.bcebos.com/rocm/release/paddlepaddle_rocm-0.0.0-cp37-cp37m-linux_x86_64.whl
    whl=`ls |grep ".whl"`
    echo $whl
    python -m pip uninstall -y paddlepaddle-rocm
    python -m pip install $whl -i https://mirror.baidu.com/pypi/simple
    tar="release"
fi

echo $tar
echo 'paddle commit:'
python -c 'import paddle; print(paddle.__version__, paddle.version.commit)'

# pytest dependency
python -m pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-assume -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-html -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-timeout -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-repeat -i https://pypi.tuna.tsinghua.edu.cn/simple
pytest -sv . --html=report_${tar}/rocm_${tar}.html --capture=tee-sys
tar cf report_${tar}.tar report_${tar}

# pytest -sv  --count=10 -x test_seg_models.py::test_fastscnn_cityscapes_1024x1024_160k
