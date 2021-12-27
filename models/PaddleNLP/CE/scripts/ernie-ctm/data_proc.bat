@echo off
cd ../..

cd models_repo\examples\text_to_knowledge\ernie-ctm\
python -m wget https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/wordtag_dataset_v2.tar.gz
tar -zxvf wordtag_dataset_v2.tar.gz
