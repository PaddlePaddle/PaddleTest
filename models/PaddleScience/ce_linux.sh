#!/bin/bash

root_dir=$PWD
cases=`find ./examples -name "*.py" | sort`

cd examples/cylinder/2d_unsteady_continuous
python download_dataset.py
cd -

ignore="log.py dataset.py download_dataset.py loading_cfd_data.py cylinder2d_unsteady_predict.py utils.py"
bug=0

export CUDA_VISIBLE_DEVICES=0
echo "" >  ${root_dir}/result.txt
echo "========= serial bug file list =========" >> ${root_dir}/result.txt
for file in ${cases}
do
echo ${file}
file_name=`basename $file`
file_dir=`dirname $file`
echo ============================= serial ${file_name} start ============================
    if [[ ${ignore} =~ ${file_name} ]]; then
        echo "skip"
    else
        cd ${file_dir}
        python ${file_name} >> ${file_name%.*}.log
        if [ $? -ne 0 ]; then
            echo ${file_name} >> ${root_dir}/result.txt
            bug=`expr ${bug} + 1`
        fi
        cat ${file_name%.*}.log
        cd -
    fi
echo ============================= serial ${file_name}  end! =============================
done

export CUDA_VISIBLE_DEVICES=0,1
ignore="log.py dataset.py download_dataset.py loading_cfd_data.py viv_inverse_predict.py viv_inverse_train.py cylinder2d_unsteady_train.py cylinder2d_unsteady_predict.py utils.py"
echo "========= distributed bug file list =========" >> ${root_dir}/result.txt
for file in ${cases}
do
echo ${file}
file_name=`basename $file`
file_dir=`dirname $file`
echo ============================= distributed ${file_name} start ============================
    if [[ ${ignore} =~ ${file_name} ]]; then
        echo "skip"
    else
        cd ${file_dir}
        python -m paddle.distributed.launch --devices=0,1 ${file_name} >> dst_${file_name%.*}.log
        if [ $? -ne 0 ]; then
            echo ${file_name} >> ${root_dir}/result.txt
            bug=`expr ${bug} + 1`
        fi
        cat dst_${file_name%.*}.log
        cd -
    fi
echo ============================= distributed ${file_name}  end! =============================
done


echo "total bug: $bug"
cat ${root_dir}/result.txt
exit ${bug}
