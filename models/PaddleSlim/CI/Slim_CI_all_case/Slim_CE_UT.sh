#!/bin/bash

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo ---${log_path}/FAIL_$2---
    echo "fail log as follow"
    cat  ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo ---${log_path}/SUCCESS_$2---
    # grep -i 'Memory Usage' ${log_path}/SUCCESS_$2.log
fi
}

cd ${slim_dir}/tests

test_num=1
static_test_num=`ls test_*.py|wc -l`
dygraph_test_num=`ls dygraph/test_*.py|wc -l`
act_test_num=`ls act/test_*.py|wc -l`
all_test_num=`expr ${static_test_num} + ${dygraph_test_num} + ${act_test_num}`

run_api_case(){
cases=`find ./ -name "test*.py" | sort`
for line in `ls test_*.py | sort`
do
    {
        name=`echo ${line} | cut -d \. -f 1`
        echo ${test_num}_"/"_${all_test_num}_${name}
        python ${line} > ${log_path}/${test_num}_${name} 2>&1
        print_info $? ${test_num}_${name}
    }&
    let test_num++
done
}
run_api_case
wait

run_api_case_dygraph(){
if [ -d ${slim_dir}/tests/dygraph ];then
cd ${slim_dir}/tests/dygraph
for line in `ls test_*.py | sort`
do
    {
    name=`echo ${line} | cut -d \. -f 1`
    echo ${test_num}_"/"_${all_test_num}_dygraph_${name}
    python ${line} > ${log_path}/${test_num}_dygraph_${name} 2>&1
    print_info $? ${test_num}_dygraph_${name}
    }&
    let test_num++
done
else
    echo -e "\033[31m no tests/dygraph \033[0m"
fi
}
run_api_case_dygraph
wait

run_api_case_act(){
if [ -d ${slim_dir}/tests/act ];then
cd ${slim_dir}/tests/act
for line in `ls test_*.py | sort`
do
    {
    name=`echo ${line} | cut -d \. -f 1`
    echo ${test_num}_"/"_${all_test_num}_act_${name}
    python ${line} > ${log_path}/${test_num}_act_${name} 2>&1
    print_info $? ${test_num}_act_${name}
    }&
    let test_num++
done
else
    echo -e "\033[31m no tests/act \033[0m"
fi
}
run_api_case_act
wait

exit $?
