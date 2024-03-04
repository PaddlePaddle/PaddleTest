#!/bin/bash

pip install -r requirements.txt

rm -rf data/
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/fastdit_features/fastdit_imagenet256.tar
tar -xvf fastdit_imagenet256.tar
rm -rf fastdit_imagenet256.tar
