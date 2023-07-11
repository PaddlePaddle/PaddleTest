
# 定义 test 文件夹的路径
path_to_api_tests="../../test/equation"

# 设置要跳过的文件名
skip_apis=("test_biharmonic.py")
error_files=()
error_num=0
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
exit ${error_num}
