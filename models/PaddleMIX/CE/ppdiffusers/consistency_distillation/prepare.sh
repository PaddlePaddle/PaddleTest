#!/bin/bash

rm -rf data
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
tar -zxvf laion400m_demo_data.tar.gz
rm -rf laion400m_demo_data.tar.gz
