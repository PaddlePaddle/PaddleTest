pip3.7 install pytest
export FLAGS_call_stack_level=2
cases=`find . -name "test*.py" | sort`
ignore="test_adaptive_avg_pool1D.py \
test_adaptive_avg_pool2D.py \
test_functional_celu.py \
test_CELU.py \
test_adaptive_avg_pool3D.py"
bug=0

echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        python3.7 -m pytest ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

echo "total bugs: "${bug} >> result.txt
exit ${bug}
