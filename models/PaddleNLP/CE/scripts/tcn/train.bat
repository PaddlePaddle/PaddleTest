@echo off
cd ../..
if not exist log\tcn md log\tcn
set logpath=%cd%\log\tcn
cd models_repo\examples\time_series\tcn
python train.py --data_path time_series_covid19_confirmed_global.csv --epochs 2 --batch_size 32 --use_gpu > %logpath%/train_%1.log 2>&1
