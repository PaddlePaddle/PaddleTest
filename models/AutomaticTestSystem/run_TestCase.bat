set sed="C:\Program Files\Git\usr\bin\sed.exe"
set CUDA_VISIBLE_DEVICES=0
python -m pip install -r requirements.txt
rem python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
python -m pytest -sv %1  --alluredir=./result
xcopy environment\environment.properties_win ./result /s /e /y /d
cd result
ren environment.properties_win environment.properties
cd ..
allure generate ./result/ -o ./report_test/ --clean
