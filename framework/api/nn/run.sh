pip3.7 install pytest
export FLAGS_call_stack_level=2
cases=`find . -name "test*.py" | sort`
ignore=""
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        python3.7 -m pytest ${file}
    fi
done
