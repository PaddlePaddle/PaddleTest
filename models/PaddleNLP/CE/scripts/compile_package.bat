@echo off
cd ..

cd models_repo

python setup.py bdist_wheel

for %%i in (".\dist\*.whl") do (
    set FileName=%%~nxi
)

python -m pip uninstall -y paddlenlp

python -m pip install dist\%FileName%
