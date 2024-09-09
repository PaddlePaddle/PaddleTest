#!/bin/bash

TOTAL_ERRORS=0


script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../../paddle_log
$script_dir/../../paddle_log

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR:$PYTHONPATH

ignore="safecheck.py"

for file in $(git diff --name-only pr_${AGILE_PULL_ID} ${AGILE_COMPILE_BRANCH}); do
    if [[ ${file} =~ ${ignore} ]]; then
        echo "ignore safecheck.py"
    else
        python ./tools/codestyle/safecheck.py ${file};
        TOTAL_ERRORS=$(expr $TOTAL_ERRORS + $?);
    fi
done

exit $TOTAL_ERRORS
