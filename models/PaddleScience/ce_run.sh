#!/bin/bash

root_dir=$PWD
cases=`find ./examples -name "*.py" | sort`

ignore=""
example_bug=0

echo "" >  ${root_dir}/result.txt
echo "========= bug file list =========" > ${root_dir}/result.txt

for file in ${cases}
do
file_name=`basename $file`
file_dir=`dirname $file`

echo ============================= ${file_name} start ============================
    if [[ ${ignore} =~ ${file_name} ]]; then
        echo "skip"
    else
        cd ${file_dir}
        python ${file_name} >> ${file_name%.*}.log
        if [ $? -ne 0 ]; then
            echo ${file_name} >> ${root_dir}/result.txt
            example_bug=`expr ${example_bug} + 1`
        fi
        cat ${file_name%.*}.log
        cd -
    fi
echo ============================= ${file_name}  end! =============================
done

#generate loss curve
cd ./examples
log_files=`find . -name "*.log" | sort`
for file in ${log_files}
do
file_name=`basename $file`

mv ${file} .
python log.py ${file_name}

done

bug=$example_bug
echo "total bug: $bug"
cat ${root_dir}/result.txt
exit ${bug}
