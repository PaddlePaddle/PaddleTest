@echo off
cd ../..

cd models_repo\applications\sentiment_analysis\

if not checkpoints/ext_checkpoints md checkpoints/ext_checkpoints
if not checkpoints/cls_checkpoints md checkpoints/cls_checkpoints
if not checkpoints/pp_checkpoints  md checkpoints/pp_checkpoints
if not exist data md data
cd data
python -m wget https://bj.bcebos.com/v1/paddlenlp/data/ext_data.tar.gz
tar -xzvf ext_data.tar.gz
python -m wget https://bj.bcebos.com/v1/paddlenlp/data/cls_data.tar.gz
tar -xzvf cls_data.tar.gz
