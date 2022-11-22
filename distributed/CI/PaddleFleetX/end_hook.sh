#!/usr/bin/env bash
export log_path=/paddle/log

function end_hook() {
    num=`cat $log_path/result.log | grep "failed" | wc -l`
    if [ "${num}" -gt "0" ];then
        echo -e "=============================base cases============================="
        cat $log_path/result.log | grep "failed"
        echo -e "===================================================================="
        exit 1
    else
        exit 0
    fi
}

main() {
    end_hook
}

main$@
