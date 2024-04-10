#! /bin/bash
# sh run_single_model.sh model_name
echo $(date +%Y-%m-%d" "%H:%M:%S) >> result_${1}.txt

export FLAGS_call_stack_level=2
cases=$(find . -name "test_${1}*.py" | sort)

# ignore="test_${1}_ort.py"
bug=0

echo "============ failed cases =============" >> result_${1}.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        python -m pytest -m server --disable-warnings -sv ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result_${1}.txt
            bug=$((bug+1))
        fi
    fi
done

echo "total bugs: "${bug} >> result_${1}.txt
exit ${bug}
