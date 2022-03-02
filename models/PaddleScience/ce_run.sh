#!/bin/bash

root_dir=$PWD
cases=`find ./examples -name "*.py" | sort`
cases1=`find ./api -name "test_*.py" | sort`

ignore=""
example_bug=0
api_bug=0

echo "" >  ${root_dir}/result.txt
echo "========= bug file list =========" > ${root_dir}/result.txt

echo "**** examples bugs ****" >> ${root_dir}/result.txt
for file in ${cases}
do
file_name=`basename $file`
file_dir=`dirname $file`

echo ${file_name}
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
echo ************************** models test finished **************************
echo "examples bug numbers: "${example_bug} >> ${root_dir}/result.txt


echo "**** api bugs ****" >> ${root_dir}/result.txt
for file in ${cases1}
do
file_name=`basename $file`
file_dir=`dirname $file`

echo ${file_name}
    if [[ ${ignore} =~ ${file_name} ]]; then
        echo "skip"
    else
	echo $PWD
	cd ${file_dir}
        python -m pytest ${file_name}
        if [ $? -ne 0 ]; then
            echo ${file_name} >> ${root_dir}/result.txt
            api_bug=`expr ${api_bug} + 1`
        fi
        cd -
    fi
echo ============================= ${file_name}  end! =============================
done
echo *************************** api test finished ****************************
echo "api bug numbers: "${api_bug} >> ${root_dir}/result.txt

bug=$[$example_bug+$api_bug]
echo "total bug: $bug"
cat ${root_dir}/result.txt
exit ${bug}
