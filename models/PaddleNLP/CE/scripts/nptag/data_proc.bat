@echo off
cd ../..

cd models_repo\examples\text_to_knowledge\nptag

python -m wget https://bj.bcebos.com/paddlenlp/paddlenlp/datasets/nptag_dataset.tar.gz
tar -zxvf nptag_dataset.tar.gz
