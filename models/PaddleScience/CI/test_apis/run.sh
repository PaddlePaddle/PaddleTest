#!/bin/bash

error_num=0

# 运行 run_api.sh
bash ./run_api.sh
exit_code=$?
error_num=$((error_num + exit_code))

# 运行 run_doctest.sh
bash ./run_doctest.sh
exit_code=$?
error_num=$((error_num + exit_code))

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