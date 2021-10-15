@echo off
cd ../..
cd models_repo\examples\model_compression\minilmv2\
wget https://paddlenlp.bj.bcebos.com/models/general_distill/minilmv2_6l_768d_ch.tar.gz --no-check-certificate
tar -zxf minilmv2_6l_768d_ch.tar.gz
