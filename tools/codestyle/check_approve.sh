#!/bin/bash

TOTAL_ERRORS=0


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR:$PYTHONPATH

ignore="safecheck.py"

for file in $(git diff --name-only HEAD HEAD^); do
    if [[ ${file} =~ ${ignore} ]]; then
        echo "ignore safecheck.py"
    else
        python ./tools/codestyle/safecheck.py ${file};
        TOTAL_ERRORS=$(expr $TOTAL_ERRORS + $?);
    fi
done

exit $TOTAL_ERRORS
