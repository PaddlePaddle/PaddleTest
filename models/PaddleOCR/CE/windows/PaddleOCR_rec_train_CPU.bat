@ echo off
rem set log_path=log
set gpu_flag=True
set sed="C:\Program Files\Git\usr\bin\sed.exe"

set log_path=log
md log
if not exist train_data (mklink /j train_data %data_path%\PaddleOCR\train_data)
if not exist pretrain_models (mklink /j pretrain_models %data_path%\PaddleOCR\pretrain_models)

rem dependency
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


setlocal enabledelayedexpansion
for /f %%i in (ocr_rec_models_list.txt) do (
rem echo %%i
rem sed -i 's!data_lmdb_release/training!data_lmdb_release/validation!g' %%i
%sed% -i s/"training"/"validation"/g %%i
set target=%%i
rem echo !target!
set target1=!target:*/=!
rem echo !target1!
set target2=!target1:*/=!
rem echo !target2!
set model=!target2:.yml=!
echo !model!
rem nvidia-smi
rem train
del %log_path%\result.log
python  tools/train.py -c %%i -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.save_model_dir="output/"!model! Train.loader.batch_size_per_card=16 > %log_path%/!model!_train.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,train,FAIL  >> %log_path%\result.log
        echo  training of !model! failed!
) else (
        echo   !model!,train,SUCCESS  >> %log_path%\result.log
        echo   training of !model! successfully!
)

rem eval
rem python tools/eval.py -c %%i -o pretrained_model="./output/!model!/0/ppcls" -o load_static_weights=False -o use_gpu=%gpu_flag% >%log_path%/!model!_eva.log 2>&1
python tools/eval.py -c %%i  -o Global.use_gpu=False Global.checkpoints="output/"!model!"/latest" > %log_path%/!model!_eval.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  evaling of !model! failed!
) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of !model! successfully!
)

rem infer
python tools/infer_rec.py -c %%i  -o Global.use_gpu=False Global.checkpoints="output/"!model!"/latest" Global.infer_img=doc/imgs_words/en/word_1.png > %log_path%/!model!_infer.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,infer,FAIL  >> %log_path%\result.log
        echo  infering of !model! failed!
) else (
        echo   !model!,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of !model! successfully!
)

rem export_model
python tools/export_model.py -c %%i -o Global.use_gpu=False Global.checkpoints="output/"!model!"/latest"  Global.save_inference_dir="./models_inference/"!model! > %log_path%/!model!_export.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,export_model,FAIL  >> %log_path%\result.log
        echo  export_modeling of !model! failed!
) else (
        echo   !model!,export_model,SUCCESS  >> %log_path%\result.log
        echo   export_model of !model! successfully!
)
rem predict
rem python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"!model! --rec_image_shape="3, 32, 100" --rec_char_type="en" > %log_path%/!model!_predict.log 2>&1
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"!model! --rec_image_shape="3, 32, 100" --rec_char_dict_path=./ppocr/utils/ic15_dict.txt > %log_path%/!model!_predict.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,predict,FAIL  >> %log_path%\result.log
        echo  predicting of !model! failed!
) else (
        echo   !model!,predict,SUCCESS  >> %log_path%\result.log
        echo   predicting of !model! successfully!
)

)

rem ********************************
rem  thge other

setlocal enabledelayedexpansion
for /f %%i in (ocr_rec_models_list_2.txt) do (
rem echo %%i
rem sed -i 's!data_lmdb_release/training!data_lmdb_release/validation!g' %%i
%sed% -i s/"training"/"validation"/g %%i
set target=%%i
rem echo !target!
set target1=!target:*/=!
rem echo !target1!
set target2=!target1:*/=!
set target2=!target2:*/=!
rem echo !target2!
set model=!target2:.yml=!
echo !model!
rem nvidia-smi
rem train
python  tools/train.py -c %%i -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.save_model_dir="output/"!model! Train.loader.batch_size_per_card=16 > %log_path%/!model!_train.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,train,FAIL  >> %log_path%\result.log
        echo  training of !model! failed!
) else (
        echo   !model!,train,SUCCESS  >> %log_path%\result.log
        echo   training of !model! successfully!
)

rem eval
rem python tools/eval.py -c %%i -o pretrained_model="./output/!model!/0/ppcls" -o load_static_weights=False -o use_gpu=%gpu_flag% >%log_path%/!model!_eva.log 2>&1
python tools/eval.py -c %%i  -o Global.use_gpu=False Global.checkpoints="output/"!model!"/latest" > %log_path%/!model!_eval.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,eval,FAIL  >> %log_path%\result.log
        echo  evaling of !model! failed!
) else (
        echo   !model!,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of !model! successfully!
)

rem infer
python tools/infer_rec.py -c %%i  -o Global.use_gpu=False Global.checkpoints="output/"!model!"/latest" Global.infer_img=doc/imgs_words/en/word_1.png > %log_path%/!model!_infer.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,infer,FAIL  >> %log_path%\result.log
        echo  infering of !model! failed!
) else (
        echo   !model!,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of !model! successfully!
)

rem export_model
python tools/export_model.py -c %%i -o Global.use_gpu=False Global.checkpoints="output/"!model!"/latest"  Global.save_inference_dir="./models_inference/"!model! > %log_path%/!model!_export.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,export_model,FAIL  >> %log_path%\result.log
        echo  export_modeling of !model! failed!
) else (
        echo   !model!,export_model,SUCCESS  >> %log_path%\result.log
        echo   export_model of !model! successfully!
)
rem predict
rem python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"!model! --rec_image_shape="3, 32, 100" --rec_char_type="en" > %log_path%/!model!_predict.log 2>&1
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"!model! --rec_image_shape="3, 32, 100" --rec_char_dict_path=./ppocr/utils/ic15_dict.txt > %log_path%/!model!_predict.log 2>&1
if not !errorlevel! == 0 (
        echo   !model!,predict,FAIL  >> %log_path%\result.log
        echo  predicting of !model! failed!
) else (
        echo   !model!,predict,SUCCESS  >> %log_path%\result.log
        echo   predicting of !model! successfully!
)
echo -----------------------------------------------------------

)

rem call PaddleOCR_det_train.bat
call PaddleOCR_det_train_CPU.bat
