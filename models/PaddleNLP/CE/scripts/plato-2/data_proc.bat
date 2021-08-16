@echo off
cd ../..

cd models_repo\examples\dialogue\plato-2\

python -m wget https://paddlenlp.bj.bcebos.com/models/transformers/plato2/data.tar.gz
tar -zxf data.tar.gz
