#!/bin/bash

bash run_api.sh 
error_num=$?
echo "===== api bug num =====" >>  result.txt
echo ${error_num} >>  result.txt

bash run_doctest.sh 
exit_code0=$?
echo "===== doctest bug num =====" >>  result.txt
echo ${exit_code0} >>  result.txt

# 运行 run_test_comments.sh
python${py_version} test_comments.py
exit_code1=$?
echo "===== comments bug num =====" >>  result.txt
echo ${exit_code1} >>  result.txt

# 运行 test_docs.py
python${py_version} test_docs.py
exit_code2=$?
echo "===== docs bug num =====" >>  result.txt
echo ${exit_code2} >>  result.txt

# 统计退出码之和
error_num=$((error_num + exit_code0 + exit_code1 + exit_code2))

# 判断退出码之和是否为零
if [ $error_num -ne 0 ]; then
    echo "One or more scripts failed with non-zero exit code"
    echo "error_num: $error_num"
fi
exit ${error_num}
