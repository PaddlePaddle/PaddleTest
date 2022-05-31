@echo off
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set root_path=%cd%
echo %root_path%

::存放 PaddleRec repo代码
if exist ./repos rd /s /q repos
mkdir repos && cd repos
set repos_path=%cd%
echo %repos_path%
cd ..

:: log文件统一存储
if exist ./logs rd /s /q logs
mkdir logs && cd logs
set log_path=%cd%
echo %log_path%
cd ..

cd %repos_path%
rd /q /s  PaddleRec
git clone --depth=10 http://github.com/PaddlePaddle/PaddleRec.git -b %1
set repo_path=%repos_path%/PaddleRec
echo ---list---
dir

call :rec_demo

rem cd %log_path%
rem for /f "delims=" %%i in (' find /C "FAIL" result.log ') do set result=%%i
rem echo %result:~-1%

rem for /f "delims=" %%i in (' echo %result:~-1% ') do set exitcode=%%i
rem echo -----fail case:%exitcode%---------
rem echo -----exit code:%exitcode%---------
rem exit %exitcode%


:rec_demo
cd %repo_path%
call :contentunderstanding_demo
rem call :match_demo
rem call :multitask_demo
rem call :rank_demo
rem call :recall_demo
rem call :recall_demo_2
goto :eof


:contentunderstanding_demo
echo ----start run contentunderstanding---
for %%I in (tagspace textcnn) do (
echo ----contentunderstanding:%%I running----
python -u tools/trainer.py -m models/contentunderstanding/%%I/config.yaml > %log_path%\%%I_demo_dy_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/contentunderstanding/%%I/config.yaml > %log_path%\%%I_demo_dy_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/contentunderstanding/%%I/config.yaml >%log_path%\%%I_demo_st_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_infer.py -m models/contentunderstanding/%%I/config.yaml >%log_path%\%%I_demo_st_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof


:match_demo
echo ----start run match----
for %%I in (dssm match-pyramid multiview-simnet) do (
echo ----match:%%I running----
python -u tools/trainer.py -m models/match/%%I/config.yaml > %log_path%\%%I_demo_dy_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/match/%%I/config.yaml > %log_path%\%%I_demo_dy_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/match/%%I/config.yaml >%log_path%\%%I_demo_st_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_infer.py -m models/match/%%I/config.yaml >%log_path%\%%I_demo_st_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

:multitask_demo
echo start run multitask
for  %%I in (esmm mmoe ple share_bottom) do (
echo ----multitask:%%I running----
python -u tools/trainer.py -m models/multitask/%%I/config.yaml > %log_path%\%%I_demo_dy_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/multitask/%%I/config.yaml > %log_path%\%%I_demo_dy_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/multitask/%%I/config.yaml > %log_path%\%%I_demo_st_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_trainer.py -m models/multitask/%%I/config.yaml > %log_path%\%%I_demo_st_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

echo 4 rank(8/21)
:rank_demo
echo start run rank
for  %%I in (deepfm dnn fm logistic_regression wide_deep gatenet xdeepfm ffm) do (
echo ----rank:%%I running----
python -u tools/trainer.py -m models/rank/%%I/config.yaml >%log_path%\%%I_demo_dy_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/rank/%%I/config.yaml >%log_path%\%%I_demo_dy_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/rank/%%I/config.yaml >%log_path%\%%I_demo_st_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_trainer.py -m models/rank/%%I/config.yaml >%log_path%\%%I_demo_st_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

echo 5 recall(1/21)
:recall_demo
echo start run recall
for  %%I in (ncf) do (
echo ----recall:%%I running----
python -u tools/trainer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_dy_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_dy_infer.log 2 >&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_st_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_infer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_st_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

python -u tools/trainer.py -m models/recall/word2vec/config.yaml
echo 5 recall(2/21)
:recall_demo_2
echo start run recall
for  %%I in (word2vec mind) do (
echo ----%%I running----
python -u tools/trainer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_dy_train.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u models/recall/%%I/infer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_dy_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_st_train.log 2>&1 
call :printInfo %errorlevel% %%I_demo_st_train
python -u models/recall/%%I/static_infer.py -m models/recall/%%I/config.yaml > %log_path%\%%I_demo_st_infer.log 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
python -u tools/trainer.py -m models/recall/word2vec/config.yaml > %log_path%\word2vec_demo_dy_train_new.log 2>&1
call :printInfo %errorlevel% word2vec_demo_dy_train_new
::python -u models/recall/%%I/infer.py -m models/recall/word2vec/config.yaml > %log_path%\word2vec_demo_dy_infer_new.log 2>&1
::call :printInfo %errorlevel% word2vec_demo_dy_infer_new
goto :eof

::########################################################################
:rec_con
echo ----------------- starting run con cpu -------------------------
echo 1 contentunderstanding (3/3)
echo 1.1 tagspace  1epoch
set model=tagspace
cd %repo_path%/models/contentunderstanding/%model%
move data data_bk
mklink /J senta_data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 1.2 textcnn
set model=textcnn
cd %repo_path%/models/contentunderstanding/%model%
move data data_bk
mklink /J senta_data %dataset_path%\rec_repo\%model%\senta_data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 1.3 textcnn_pretrain
set model=textcnn_pretrain
cd %repo_path%/models/contentunderstanding/%model%
move data data_bk
mklink /J senta_data %dataset_path%\rec_repo\%model%\senta_data
mklink /J pretrain_model %dataset_path%\rec_repo\%model%\pretrain\pretrain_model
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 2 match(3/3) not

echo 3.1 multitask (2/4)  esmm
set model=esmm
cd %repo_path%/models/rank/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 3.2 mmoe
set model=mmoe
cd %repo_path%/models/rank/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 4.1 rank(5/21) deepfm
set model=deepfm
cd %repo_path%/models/rank/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 4.2 dnn
set model=dnn
cd %repo_path%/models/rank/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 4.3 fm
set model=fm
cd %repo_path%/models/rank/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 4.4 lr
set model=logistic_regression
cd %repo_path%/models/rank/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 4.5 wide_deep
set model=wide_deep
cd %repo_path%/models/rank/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo  5 recall (3/8)
echo  5.1 gnn
set model=gnn
cd %repo_path%/models/recall/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%

echo 5.2 word2vec
set model=word2vec
cd %repo_path%/models/recall/%model%
move data data_bk
mklink /J data %dataset_path%\rec_repo\%model%\data
call :run_con_cpu %model%
call :run_con_gpu %model%
python infer.py --test_dir ./data/all_test --dict_path ./data/all_dict/word_id_dict.txt ^
--batch_size 10000 --model_dir ./increment_w2v_cpu/  ^
--start_index 0 --last_index 4 --emb_size 300 >${log_path}/%model%_infer_cpu.log 2>&1
call :printInfo %errorlevel% %model%_infer_cpu
python infer.py --test_dir ./data/all_test --dict_path ./data/all_dict/word_id_dict.txt ^
--batch_size 10000 --model_dir ./increment_w2v_gpu/  ^
--start_index 0 --last_index 4 --emb_size 300 >${log_path}/%model%_infer_gpu1.log 2>&1
call :printInfo %errorlevel% %model%_infer_gpu1

echo 5.3 youtube_dnn
set model=youtube_dnn
cd %repo_path%/models/recall/%model%
::需要转成Python的,或者是直接软连数据，目前是全量的数据集还比较大
::sh data_prepare.sh
call :run_con_cpu %model%
call :run_con_gpu %model%
python infer.py --test_epoch 19 --inference_model_dir ./inference_youtubednn_cpu ^
--increment_model_dir ./increment_youtubednn_cpu --watch_vec_size 64 ^
--search_vec_size 64 --other_feat_size 64 --topk 5 > ${log_path}/%model%_infer_cpu.log 2>&1
call :printInfo %errorlevel% %model%_infer_cpu
python infer.py --use_gpu 1 --test_epoch 19 ^
--inference_model_dir ./inference_youtubednn_gpu ^
--increment_model_dir ./increment_youtubednn_gpu ^
--watch_vec_size 64 --search_vec_size 64 ^
--other_feat_size 64 --topk 5 >${log_path}/%model%_infer_gpu1.log 2>&1
call :printInfo %errorlevel% %model%_infer_gpu1
echo ----------------- run con cpu end -------------------------
goto :eof

::########################################################################

:printInfo
if %1 == 0 (
    move %log_path%\%2.log %log_path%\SUCCESS_%2.log
    echo SUCCESS_%2.log
    echo SUCCESS_%2.log >> %log_path%\result.log
) else (
    move %log_path%\%2.log %log_path%\FAIL_%2.log
    echo  FAIL_%2.log
    type %log_path%\FAIL_%2.log
    echo  FAIL_%2.log >> %log_path%\result.log
)
goto :eof


:run_con_cpu
copy %dataset_path%\rec_repo\rec_config\%1_cpu_config.yaml .
echo start run cpu : python -m paddlerec.run -m %1_cpu_config.yaml
python -m paddlerec.run -m %1_cpu_config.yaml >%log_path%\%1_cpu 2>&1
call :printInfo %errorlevel% %1_cpu
goto :eof

:run_con_gpu
copy %dataset_path%\rec_repo\rec_config\%1_gpu_config.yaml .
echo start run gpu : python -m paddlerec.run -m %1_gpu_config.yaml
python -m paddlerec.run -m %1_gpu_config.yaml >%log_path%\%1_gpu1 2>&1
call :printInfo %errorlevel% %1_gpu1
goto :eof
