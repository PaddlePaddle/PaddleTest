@echo off
cd ..
@REM python -m pip uninstall --user Pillow -y
@REM python -m pip install --user Pillow==8.4.0
cd models_repo

python setup.py bdist_wheel

for %%i in (".\dist\*.whl") do (
    set FileName=%%~nxi
)

python -m pip uninstall -y paddlenlp
python -m pip install dist\%FileName%
