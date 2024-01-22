#!/bin/bash

error_num=0

echo "===== doctest bug list =====" >  result.txt
while IFS= read -r -d '' file; do
    if [[ "${skip_files[@]}" =~ "${file##*/}" ]]; then
        echo "Skipping ${file} ..."
    else
        echo "Running doctest on ${file} ..."
        python${py_version} -m doctest "${file}"
        if [ $? -ne 0 ]; then
            error_files+=("${file}")
            error_num=$((error_num + 1))
            (echo "Error files:"; echo "${file}") >> result.txt
        fi
    fi
done < <(find "$path_to_ppsci" -type f -name "*.py" -print0)

# 定义 ppsci 文件夹的路径
path_to_api_tests="../../test/equation"

# 设置要跳过的文件名
skip_apis=("test_biharmonic.py")

# 遍历 ppsci 文件夹下所有的 Python 文件，统计退出码不为零的文件名
echo "===== apis bug list =====" >>  result.txt
while IFS= read -r -d '' file; do
    if [[ "${skip_apis[@]}" =~ "${file##*/}" ]]; then
        echo "Skipping ${file} ..."
    else
        echo "Running apitest on ${file} ..."
        python${py_version} -m pytest "${file}"
        if [ $? -ne 0 ]; then
            error_files+=("${file}")
            error_num=$((error_num + 1))
            (echo "Error apis:"; echo "${file}") >> result.txt
        fi
    fi
done < <(find "$path_to_api_tests" -type f -name "test_*.py" -print0)

if [ ${#error_files[@]} -gt 0 ]; then
    echo "The following files test failed:"
    echo ${error_num}
    printf '%s\n' "${error_files[@]}"
else
    echo "All files passed ."
fi

# 运行 run_test_comments.sh
bash ./run_test_comments.sh
exit_code=$?
error_num=$((error_num + exit_code))

# 判断退出码之和是否为零
if [ $error_num -ne 0 ]; then
    echo "One or more scripts failed with non-zero exit code"
    echo "error_num: $error_num"
fi
exit ${error_num}
