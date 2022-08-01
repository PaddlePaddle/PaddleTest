#!/usr/bin/env bash

# 批量调整Det下所有yml中学习率
yml_list=`find Det -name "*.yml" | sort`
for yml in ${yml_list}
do
echo ${yml}
sed -i "" "s/dtype: \"float32\"/dtype: \"float64\"/g" ${yml}
done
