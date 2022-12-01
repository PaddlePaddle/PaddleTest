print_info(){
if [ $1 -ne 0 ];then
    mv ${nlp_dir}/unittest_logs/$2.log ${nlp_dir}/unittest_logs/$2_FAIL.log
    echo -e "\033[31m ${nlp_dir}/unittest_logs/$2_FAIL \033[0m"
    cat ${nlp_dir}/unittest_logs/$2_FAIL.log
else
    echo -e "\033[32m ${unittest_path}/$2_SUCCESS \033[0m"
fi
}
transformers(){
# RUN all transformers unitests
cd ${nlp_dir}/tests/transformers/
for apicase in `ls`;do
    if [[ ${apicase##*.} == "py" ]];then
            continue
    else
        pytest tests/transformers/${apicase}/test_*.py  >${nlp_dir}/unittest_logs/${apicase}_unittest.log 2>&1
        print_info $? ${apicase}_unittest
    fi
done
}
$1