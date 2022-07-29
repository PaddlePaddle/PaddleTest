#!/usr/bin/env bash

# 批量调整Det下所有yml中学习率
yml_list=`find Det -name "*.yml" | sort`
for yml in ${yml_list}
do
echo ${yml}
sed -i "" "s/learning_rate: 0.000001/learning_rate: 0.00001/g" ${yml}
done
