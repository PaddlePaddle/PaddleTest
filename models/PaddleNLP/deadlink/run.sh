#!/usr/bin/env bash
echo "dead link check"
set +x
root_path=$PWD
echo $root_path
#export PATH=/usr/local/python3/bin:$PATH
# get_base_code
REPO=${1:-"PaddleNLP"}
BRANCH=${2:-"develop"}
email_addr=${3:-"zhuweiguo@baidu.com,zhangjunjun04@baidu.com,liujie44@baidu.com"}
email_sub=${4:-"PaddleNLP死链检测汇总报告"}
#################################
#sh check_changed_link.sh ${REPO} ${BRANCH}
python test_deadlink.py --code_path ${REPO} --repo ${REPO} --branch ${BRANCH} --func all;
set -x

#检查程序是否执行成功
if [ ! -f "${REPO}_${BRANCH}_md_result.txt" ]
then
    echo "run fail"
    exit 1
fi
# send emails
python parse_result.py --data_path=$root_path/${REPO}_${BRANCH}_md_result.txt --email_addr=${email_addr} --email_sub=${email_sub}
#输出文件，并判断是否有死链（404）
mkdir result
cd result
cp -r $root_path/${REPO}_${BRANCH}_md_result* ./
NUM="$(grep 404 ${REPO}_${BRANCH}_md_result.txt | wc -l)"
FAILLINK="$(grep 404 ${REPO}_${BRANCH}_md_result.txt )"
if [ "${FAILLINK}" -gt "0" ]
then
    echo "FAIL_NUM: " ${NUM} " links"
    echo "FAIL_LINK: " ${FAILLINK} 
    exit 2
else
    echo "SUCCESS"
    exit 0
fi
