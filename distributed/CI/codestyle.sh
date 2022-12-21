echo "========InfoSafeCheck start========"
set +e

export REPO_ROOT=$PWD
exit_code=0
unset http_proxy && unset https_proxy
pip install -i http://pip.baidu.com/root/baidu/+simple/ --trusted-host pip.baidu.com github_info_leak_agent
for file_name in $(git diff --name-only pr_${AGILE_PULL_ID} ${AGILE_COMPILE_BRANCH});do
    real_filepath="${REPO_ROOT}/${file_name}"
    if [ ! -f "${real_filepath}" ];then
        echo "====skip deleted files: ${file_name}===="
        continue
    else
        echo "====${file_name}===="
        github-info-leak-agent scan --path=${real_filepath}
        if [[ "$?" != "0" ]];then
            exit_code=4
        fi
    fi
done

commit_files=on
export http_proxy=${proxy}
export https_proxy=${proxy}
for file_name in $(git diff --name-only pr_${AGILE_PULL_ID} ${AGILE_COMPILE_BRANCH});do
    real_filepath="${REPO_ROOT}/${file_name}"
    if [ ! -f "${real_filepath}" ];then
        echo "====skip deleted files: ${file_name}===="
        continue
    else
        echo "====${file_name}===="
        if ! pre-commit run --files $real_filepath; then
            commit_files=off
        fi
    fi
done

set -x
if [ $commit_files == 'off' ];then
    echo "code format error"
    git diff 2>&1
    exit_code=4
fi

if [[ "$exit_code" != "0" ]];then
	echo "InfoSafeCheck failed"
fi
exit $exit_code
echo "========InfoSafeCheck end========"
