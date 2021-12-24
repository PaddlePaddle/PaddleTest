@echo off
cd ..

cd models_repo

del /s /q  C:\Python39\Lib\site-packages\~umpy-1.21.5.dist-info

python setup.py bdist_wheel

for %%i in (".\dist\*.whl") do (
    set FileName=%%~nxi
)

python -m pip uninstall -y paddlenlp

python -m pip install dist\%FileName%
