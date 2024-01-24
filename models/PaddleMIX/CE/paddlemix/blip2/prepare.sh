#!/bin/bash

rm -rf /root/.paddlemix/datasets/*
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/blip2/coco.tar.gz -P /root/.paddlemix/datasets/
rm -rf /root/.paddlemix/datasets/coco_karpathy_test_gt.json
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/blip2/coco_karpathy_test_gt.json -P /root/.paddlemix/datasets/
cd /root/.paddlemix/datasets/
tar -zxvf coco.tar.gz
rm -rf coco.tar.gz

apt-get install -y default-jre
