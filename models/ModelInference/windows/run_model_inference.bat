echo 'paddle commit:'
python -c "import paddle; print(paddle.__version__, paddle.version.commit)"

rem pytest dependency
python -m pip install --upgrade pip
python -m pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-assume -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-html -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-timeout -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pytest-repeat -i https://pypi.tuna.tsinghua.edu.cn/simple

rem python -m pytest -sv . --html=report/model_inference.html --capture=tee-sys
