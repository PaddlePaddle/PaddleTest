pip3.7 install pytest
export FLAGS_call_stack_level=2
cases=`find . -name "test*.py" | sort`
ignore="test_all.py \
test_allclose.py \
test_equal.py \
test_equal_all.py \
test_greater_equal.py \
test_greater_than.py \
test_less_equal.py \
test_less_than.py \
test_logical_and.py \
test_logical_not.py \
test_logical_or.py \
test_logical_xor.py \
test_not_equal.py"
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
