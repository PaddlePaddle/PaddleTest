@echo off
cd ../..
cd models_repo\examples\semantic_indexing\
python -m pip install hnswlib
python -m pip install wget
python -m wget https://paddlenlp.bj.bcebos.com/models/semantic_index/semantic_pair_train.tsv
python -m wget https://paddlenlp.bj.bcebos.com/models/semantic_index/same_semantic.tsv
python -m wget https://paddlenlp.bj.bcebos.com/models/semantic_index/corpus_file
