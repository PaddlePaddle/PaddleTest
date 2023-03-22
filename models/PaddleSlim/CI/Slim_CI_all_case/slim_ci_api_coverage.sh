#!/usr/bin/env bash
echo "enter slim_ci_api_coverage, params:" $1,$2

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/$2_FAIL.log
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
else
    mv ${log_path}/$2 ${log_path}/$2_SUCCESS.log
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
cudaid1=$1;
cudaid2=$2;
echo "cudaid1,cudaid2", ${cudaid1}, ${cudaid2}
export CUDA_VISIBLE_DEVICES=${cudaid1}

# install lcov
curl -o /lcov-1.14.tar.gz -s https://paddle-ci.gz.bcebos.com/coverage%2Flcov-1.14.tar.gz
tar -xf /lcov-1.14.tar.gz -C /
cd /lcov-1.14
make install

cd ${slim_dir}/tests
pip install coverage

export COVERAGE_FILE=`pwd`/python-coverage.data
source=${slim_dir}

PADDLE_ROOT=${slim_dir}
test_num=1
static_test_num=`ls test_*.py|wc -l`
dygraph_test_num=`ls dygraph/test_*.py|wc -l`
act_test_num=`ls act/test_*.py|wc -l`
quant_analysis_test_num=`ls quant_analysis/test_*.py|wc -l`
quantization_test_num=`ls quantization/test_*.py|wc -l`
distribution_test_num=`ls distribution/test_*.py|wc -l`

all_test_num=`expr ${static_test_num} + ${dygraph_test_num} + ${act_test_num} + ${quant_analysis_test_num} \
    + ${quantization_test_num} + ${distribution_test_num} `

run_distribution_case(){
cd ${slim_dir}/tests/distribution
ignore=""
for line in `ls test_*.py | sort`
do
    {
    name=`echo ${line} | cut -d \. -f 1`
    echo ${test_num}_"/"_${all_test_num}_${name}
    if [[ ${ignore} =~ ${line##*/} ]]; then
        echo "skip" ${line##*/}
    else
        python -m coverage run --source=${source} --branch -p \
            -m paddle.distributed.launch ${line} > ${log_path}/${test_num}_distribution_${name} 2>&1
       print_info $? ${test_num}_distribution_${name}
    fi
    }&
    let test_num++
done
}
export CUDA_VISIBLE_DEVICES=${cudaid2}
echo "---distribution UT case is running with ${CUDA_VISIBLE_DEVICES}"
run_distribution_case
wait

export CUDA_VISIBLE_DEVICES=${cudaid1}
echo "---single card UT case is running with ${CUDA_VISIBLE_DEVICES}"
run_api_case(){
cases=`find ./ -name "test*.py" | sort`
#ignore="test_analysis_helper.py"
ignore=""
for line in `ls test_*.py | sort`
do
    {
    name=`echo ${line} | cut -d \. -f 1`
    echo ${test_num}_"/"_${all_test_num}_${name}
    if [[ ${ignore} =~ ${line##*/} ]]; then
        echo "skip" ${line##*/}
    else
       python -m coverage run --source=${source} --branch -p ${line} > ${log_path}/${test_num}_${name} 2>&1
       print_info $? ${test_num}_${name}
    fi
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
    python -m coverage run --source=${source} --branch -p ${line} > ${log_path}/${test_num}_dygraph_${name} 2>&1
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
    python -m coverage run --source=${source} --branch -p ${line} > ${log_path}/${test_num}_act_${name} 2>&1
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

run_api_case_quant_analysis(){
if [ -d ${slim_dir}/tests/quant_analysis ];then
cd ${slim_dir}/tests/quant_analysis
for line in `ls test_*.py | sort`
do
    {
    name=`echo ${line} | cut -d \. -f 1`
    echo ${test_num}_"/"_${all_test_num}_quant_analysis_${name}
    python -m coverage run --source=${source} --branch -p ${line} > ${log_path}/${test_num}_quant_analysis_${name} 2>&1
    print_info $? ${test_num}_quant_analysis_${name}
    }
    let test_num++
done
else
    echo -e "\033[31m no tests/quant_analysis \033[0m"
fi
}
run_api_case_quant_analysis

run_api_case_quantization(){
if [ -d ${slim_dir}/tests/quantization ];then
cd ${slim_dir}/tests/quantization
for line in `ls test_*.py | sort`
do
    {
    name=`echo ${line} | cut -d \. -f 1`
    echo ${test_num}_"/"_${all_test_num}_quantization_${name}
    python -m coverage run --source=${source} --branch -p ${line} > ${log_path}/${test_num}_quantization_${name} 2>&1
    print_info $? ${test_num}_quantization_${name}
    }&
    let test_num++
done
else
    echo -e "\033[31m no tests/quantization \033[0m"
fi
}
run_api_case_quantization
wait

cd ${slim_dir}/tests

coverage combine `ls python-coverage.data.*`
set -x
coverage xml -i -o python-coverage.xml
python ${PADDLE_ROOT}/coverage/python_coverage.py > python-coverage.info
# step1
function gen_python_full_html_report_extract() {
    echo -e "\033[35m ---- gen_python_full_html_report_extract \033[0m"
    lcov --extract python-coverage.info \
        '/*/paddleslim/*' \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0
    mv -f python-coverage-full.tmp python-coverage-full.info
}
# step2 optional
function gen_python_full_html_report_remove() {
    echo -e "\033[35m ---- gen_python_full_html_report_remove\033[0m"
    lcov --remove python-coverage-full.info \
        '/*/tests/*' \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0
    mv -f python-coverage-full.tmp python-coverage-full.info
}
function gen_python_full_html_report_all() {
    echo -e "\033[35m ---- gen_python_full_html_report_all\033[0m"
    genhtml -o python-coverage-full \
        -t 'Python full Coverage' \
        --no-function-coverage \
        --no-branch-coverage \
        --ignore-errors source \
        python-coverage-full.info
}
gen_python_full_html_report_extract || true
gen_python_full_html_report_remove || true
gen_python_full_html_report_all || true

# python diff html report
function gen_python_diff_html_report() {
    if [ "${GIT_PR_ID}" != "" ]; then
        COVERAGE_DIFF_PATTERN=`python ${PADDLE_ROOT}/coverage/pull_request.py files ${GIT_PR_ID}`
        echo -e "\033[35m ----COVERAGE_DIFF_PATTERN \033[0m" ${COVERAGE_DIFF_PATTERN}
        python ${PADDLE_ROOT}/coverage/pull_request.py diff ${GIT_PR_ID} > python-git-diff.out
    fi
    lcov --extract python-coverage-full.info \
        ${COVERAGE_DIFF_PATTERN} \
        -o python-coverage-diff.info \
        --rc lcov_branch_coverage=0

    echo -e "\033[35m ---- start coverage_diff.py:Removes rows that are not related to PR \033[0m"
    python ${PADDLE_ROOT}/coverage/coverage_diff.py python-coverage-diff.info python-git-diff.out '/workspace/' > python-coverage-diff.tmp
    mv -f python-coverage-diff.tmp python-coverage-diff.info

    echo -e "\033[35m ---- genhtml python-coverage-diff \033[0m"
    genhtml -o python-coverage-diff \
        -t 'Python Diff Coverage' \
        --no-function-coverage \
        --no-branch-coverage \
        --ignore-errors source \
        python-coverage-diff.info
}
gen_python_diff_html_report || true

# assert coverage lines
echo -e "\033[35m ---- Assert Python Diff Coverage \033[0m"
python ${PADDLE_ROOT}/coverage/coverage_lines.py python-coverage-diff.info 0.9 || PYTHON_COVERAGE_LINES_ASSERT=1

if [ "$PYTHON_COVERAGE_LINES_ASSERT" = "1" ]; then
    echo -e "\033[31m ---- coverage-diff < 0.9 \033[0m"
    exit 9
fi
cd ${slim_dir}/logs
ls *_FAIL*
if [ $? -eq 0 ];then
   exit 1
fi
