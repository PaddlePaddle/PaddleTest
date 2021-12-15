@echo off
set python_version="3"
set PATH=C:\Python38;C:\Python38\Lib;C:\Python38\Scripts;%PATH%
::set FLAGS_eager_delete_tensor_gb=0.0;
::set FLAGS_fast_eager_deletion_mode=1;
::set FLAGS_fraction_of_gpu_memory_to_use=0.8;
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set http_proxy=http://172.19.57.45:3128
set https_proxy=http://172.19.57.45:3128
::set PATH=C:\Python37;C:\Python37\Lib;C:\Python37\Scripts;%PATH%
python -c "import sys; print(sys.version_info[:])"

set repos=D:\ce_data\yanmeng02\rec\rec_demo
::set dataset_path=D:\guomengmeng01\all_data
set log_path=D:\ce_data\yanmeng02\rec\rec_log
::########################################################################
::注意软链的时候路径的分割是\,cd 路径的时候可以是/
::set rec_branch=develop
set rec_branch=master

echo git clone : %rec_branch%
cd %repos%

rd /q /s  PaddleRec
git clone -b %rec_branch% http://github.com/PaddlePaddle/PaddleRec.git
::git clone -b %rec_branch% https://github.com/frankwhzhang/PaddleRec.git
set repo_path=%repos%\PaddleRec

::########################################################################
::选择调用的函数
set result_time=%date:~0,4%%date:~5,2%%date:~8,2%%time:~0,2%%time:~3,2%%time:~6,2%
mkdir %log_path%\rec_logs_%result_time%
set log_path_rec=%log_path%\rec_logs_%result_time%
call :rec_demo
move %log_path_rec% %log_path_rec%_demo

::set result_time=%date:~0,4%%date:~5,2%%date:~8,2%%time:~0,2%%time:~3,2%%time:~6,2%
::mkdir %log_path%\rec_logs_%result_time%
::set log_path_rec=%log_path%\rec_logs_%result_time%
::call :rec_con $1
::move %log_path_rec% %log_path_rec%_con
goto :eof

::########################################################################

:rec_demo
cd %repo_path%
call :contentunderstanding_demo
call :match_demo
call :multitask_demo
call :rank_demo
call :recall_demo
call :recall_word2vec
goto :eof

:contentunderstanding_demo
echo start run contentunderstanding
for  %%I in (tagspace textcnn) do (
python -u tools/trainer.py -m models/contentunderstanding/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_train 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/contentunderstanding/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_infer 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/contentunderstanding/%%I/config.yaml >%log_path_rec%\%%I_demo_st_train 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_infer.py -m models/contentunderstanding/%%I/config.yaml >%log_path_rec%\%%I_demo_st_infer 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

echo 2 match(3/3)
:match_demo
echo start run match
for  %%I in (dssm match-pyramid multiview-simnet) do (
python -u tools/trainer.py -m models/match/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_train 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/match/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_infer 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/match/%%I/config.yaml >%log_path_rec%\%%I_demo_st_train 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_infer.py -m models/match/%%I/config.yaml >%log_path_rec%\%%I_demo_st_infer 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

echo 3 multitask (4/4)
:multitask_demo
echo start run multitask
for  %%I in (esmm mmoe ple share_bottom) do (
python -u tools/trainer.py -m models/multitask/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_train 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/multitask/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_infer 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/multitask/%%I/config.yaml >%log_path_rec%\%%I_demo_st_train 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_trainer.py -m models/multitask/%%I/config.yaml >%log_path_rec%\%%I_demo_st_infer 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

echo 4 rank(8/21)
:rank_demo
echo start run rank
for  %%I in (deepfm dnn fm logistic_regression wide_deep gateDnn xdeepfm ffm) do (
python -u tools/trainer.py -m models/rank/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_train 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/rank/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_infer 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/rank/%%I/config.yaml >%log_path_rec%\%%I_demo_st_train 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_trainer.py -m models/rank/%%I/config.yaml >%log_path_rec%\%%I_demo_st_infer 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

echo 5 recall(3/21)
:recall_demo
echo start run recall
for  %%I in (word2vec ncf mind) do (
python -u tools/trainer.py -m models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_train 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u tools/infer.py -m models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_infer 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo_st_train 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u tools/static_infer.py -m models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo_st_infer 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
goto :eof

echo 5 recall_word2vec
:recall_word2vec
echo start run word2vec
for  %%I in (word2vec) do (
python -u tools/trainer.py -m %repo_path%/models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_train 2>&1
call :printInfo %errorlevel% %%I_demo_dy_train
python -u %repo_path%/models/recall/%%I/infer.py -m %repo_path%/models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo 2>&1
call :printInfo %errorlevel% %%I_demo_dy_infer

python -u tools/static_trainer.py -m %repo_path%/models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo_dy_train 2>&1
call :printInfo %errorlevel% %%I_demo_st_train
python -u %repo_path%/models/recall/%%I/static_infer.py -m %repo_path%/models/recall/%%I/config.yaml >%log_path_rec%\%%I_demo 2>&1
call :printInfo %errorlevel% %%I_demo_st_infer
)
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
--start_index 0 --last_index 4 --emb_size 300 >${log_path_rec}/%model%_infer_cpu 2>&1
call :printInfo %errorlevel% %model%_infer_cpu
python infer.py --test_dir ./data/all_test --dict_path ./data/all_dict/word_id_dict.txt ^
--batch_size 10000 --model_dir ./increment_w2v_gpu/  ^
--start_index 0 --last_index 4 --emb_size 300 >${log_path_rec}/%model%_infer_gpu1 2>&1
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
--search_vec_size 64 --other_feat_size 64 --topk 5 >${log_path_rec}/%model%_infer_cpu 2>&1
call :printInfo %errorlevel% %model%_infer_cpu
python infer.py --use_gpu 1 --test_epoch 19 ^
--inference_model_dir ./inference_youtubednn_gpu ^
--increment_model_dir ./increment_youtubednn_gpu ^
--watch_vec_size 64 --search_vec_size 64 ^
--other_feat_size 64 --topk 5 >${log_path_rec}/%model%_infer_gpu1 2>&1
call :printInfo %errorlevel% %model%_infer_gpu1
echo ----------------- run con cpu end -------------------------
goto :eof

::########################################################################
:printInfo
echo "%1, %2:" %1, %2
if %1 == 1 (
    move %log_path_rec%\%2 %log_path_rec%\FAIL_%2.log
    echo  FAIL_%2.log >> %log_path_rec%\result_%result_time%.log
) else (
    move %log_path_rec%\%2 %log_path_rec%\SUCCESS_%2.log
    echo SUCCESS_%2.log >> %log_path_rec%\result_%result_time%.log
)
goto :eof

:run_con_cpu
copy %dataset_path%\rec_repo\rec_config\%1_cpu_config.yaml .
echo start run cpu : python -m paddlerec.run -m %1_cpu_config.yaml
python -m paddlerec.run -m %1_cpu_config.yaml >%log_path_rec%\%1_cpu 2>&1
call :printInfo %errorlevel% %1_cpu
goto :eof

:run_con_gpu
copy %dataset_path%\rec_repo\rec_config\%1_gpu_config.yaml .
echo start run gpu : python -m paddlerec.run -m %1_gpu_config.yaml
python -m paddlerec.run -m %1_gpu_config.yaml >%log_path_rec%\%1_gpu1 2>&1
call :printInfo %errorlevel% %1_gpu1
goto :eof
::########################################################################
