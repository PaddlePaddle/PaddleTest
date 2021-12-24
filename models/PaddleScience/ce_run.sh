#!/bin/bash

cases=`find . -name "*.py" | sort`
ignore="darcy2d.py"
bug=0

echo "" >  result.txt
for file in ${cases}
do
file_name=`basename $file`
file_dir=`dirname $file`

echo ${file_name}
    if [[ ${ignore} =~ ${file_name} ]]; then
        echo "skip"
    else
        cd ${file_dir}
        python ${file_name} --alluredir=report
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
        cd -
    fi
done
echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}
