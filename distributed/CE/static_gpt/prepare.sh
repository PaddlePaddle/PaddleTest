#!/usr/bin/env bash
cd ../
if [[ -d static_gpt.tar.gz ]];then
    rm -rf .static_gpt.tar.gz
    tar -zcvf static_gpt.tar.gz static_gpt/
else
    tar -zcvf static_gpt.tar.gz static_gpt/
fi
hadoop fs -D fs.default.name=afs://yinglong.afs.baidu.com:9902 -D hadoop.job.ugi=paddle,dltp_paddle@123 -rm /user/paddle/liujie44/static_gpt.tar.gz
hadoop fs -D fs.default.name=afs://yinglong.afs.baidu.com:9902 -D hadoop.job.ugi=paddle,dltp_paddle@123 -put static_gpt.tar.gz /user/paddle/liujie44/


