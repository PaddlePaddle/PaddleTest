echo "============================================================"
echo "PP-DataClean test start"
echo "============================================================"
set bug=0
cd ..
easydata --model image_orientation --input ./demo/image_orientation/1.jpg --device gpu
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

easydata --model image_orientation --input ./demo/image_orientation/ --device gpu
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

easydata --model image_orientation --input ./demo/image_orientation/ --device gpu --thresh 0.93
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

easydata --model clarity_assessment --input ./demo/clarity_assessment/ --device gpu
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

easydata --model code_exists --input ./demo/code_exists/ --device gpu
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

move tests\test_PPDC.py .
python test_PPDC.py
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

echo "============================================================"
echo "PP-DataAug test start"
echo "============================================================"

easydata --model ppdataaug --ori_data_dir demo/clas_data/ --label_file demo/clas_data/train_list.txt --gen_mode img2img
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

easydata --model ppdataaug --ori_data_dir demo/ocr_data/ --label_file demo/ocr_data/train_list.txt --gen_mode img2img --model_type ocr_rec
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

easydata --model ppdataaug --bg_img_dir demo/ocr_rec/bg --corpus_file demo/ocr_rec/corpus.txt --gen_mode text2img --model_type ocr_rec
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

easydata --model ppdataaug --ori_data_dir demo/shitu_data --label_file demo/shitu_data/train_list.txt --gen_mode img2img --use_big_model False
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

move tests\test_PPDA.py .
python test_PPDA.py
if %errorlevel%==0 (echo successfully) else (set /a bug=%bug%+1)

exit /b %bug%