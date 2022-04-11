@echo off

echo ----install paddle-------
python -m pip uninstall paddlepaddle-gpu -y
python -m pip uninstall paddlepaddle -y
python -m pip install %1 --no-cache-dir
echo ----paddle commit:----
python -c "import paddle; print(paddle.__version__,paddle.version.commit)";

echo git clone slim branch: %2
cd %repo_path%
:: git config --global http.sslVerify "false"
git clone https://github.com/PaddlePaddle/PaddleSlim.git -b %2

echo ----install paddleslim-------
python -m pip uninstall paddleslim -y
cd %repo_path%/PaddleSlim
git branch
python -m pip install -r requirements.txt
python setup.py install
echo --------pip list---------
python -m pip list 
