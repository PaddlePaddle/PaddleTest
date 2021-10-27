@echo off
cd ../..
cd models_repo\examples\model_compression\minilmv2\
python -m wget https://paddlenlp.bj.bcebos.com/models/general_distill/minilmv2_6l_768d_ch.tar.gz
tar -zxf minilmv2_6l_768d_ch.tar.gz
