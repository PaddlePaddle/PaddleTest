#!/bin/bash

pip install -r requirements.txt
pip uninstall ppdiffusers -y
cd ../../
pip install -e .
cd -

export HF_ENDPOINT=https://hf-mirror.com
python download_sd15.py

cd ../../examples/dreambooth
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/dogs.tar.gz
tar -zxvf dogs.tar.gz
rm -rf dogs.tar.gz
cd -

cd ../../examples/textual_inversion
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/cat-toy.tar.gz
tar -zxvf cat-toy.tar.gz
rm -rf cat-toy.tar.gz
cd -

cd ../../examples/text_to_image_laion400m
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
cp laion400m_demo_data.tar.gz ../stable_diffusion
cp laion400m_demo_data.tar.gz ../autoencoder/vae
cp laion400m_demo_data.tar.gz ../consistency_distillation/lcm_trainer
cp laion400m_demo_data.tar.gz ../ip_adapter
rm -rf laion400m_demo_data.tar.gz
cd -

cd ../../examples/stable_diffusion
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
cd -

cd ../../examples/autoencoder/vae
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
cd -

cd ../../examples/controlnet
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/fill50k.zip
unzip -o fill50k.zip
rm -rf fill50k.zip
cd -

cd ../../examples/t2i-adapter
wget https://paddlenlp.bj.bcebos.com/models/community/westfish/t2i-adapter/openpose_data_demo.tar.gz
tar -zxvf openpose_data_demo.tar.gz
rm -rf openpose_data_demo.tar.gz
cd -

cd ../../examples/consistency_distillation/lcm_trainer
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
cd -


cd ../../examples/ip_adapter
rm -rf data
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
cd -