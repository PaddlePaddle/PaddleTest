#!/bin/bash

root_dir=$PWD
cases=`find . -name "*.py" | sort`
ignore="ce_win.py"
bug=0

echo "" >  ${root_dir}/result.txt
echo "========= bug file list =========" > ${root_dir}/result.txt
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
            echo ${file_name} >> ${root_dir}/result.txt
            bug=`expr ${bug} + 1`
        fi
        cd -
    fi
echo ============================= ${file_name}  end! =============================
done
echo "total bugs: "${bug} >> ${root_dir}/result.txt
cat ${root_dir}/result.txt
exit ${bug}
