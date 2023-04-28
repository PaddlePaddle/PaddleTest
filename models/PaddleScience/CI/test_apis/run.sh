#!/bin/bash

# 定义 ppsci 文件夹的路径
path_to_ppsci="../../ppsci"

# 设置要跳过的文件名
skip_files=("inflation.py")

# 遍历 ppsci 文件夹下所有的 Python 文件，统计退出码不为零的文件名
error_files=()
error_num=0
echo "===== apis bug list =====" >  result.txt
while IFS= read -r -d '' file; do
    if [[ "${skip_files[@]}" =~ "${file##*/}" ]]; then
        echo "Skipping ${file} ..."
    else
        echo "Running doctest on ${file} ..."
        python3.7 -m doctest "${file}"
        if [ $? -ne 0 ]; then
            error_files+=("${file}")
            error_num=$((error_num + 1))
            (echo "Error files:"; echo "${file}") >> result.txt
        fi
    fi
done < <(find "$path_to_ppsci" -type f -name "*.py" -print0)


if [ ${#error_files[@]} -gt 0 ]; then
    echo "The following files exited with non-zero status:"
    echo ${error_num}
    printf '%s\n' "${error_files[@]}"
else
    echo "All files passed."
fi
exit ${error_num}
