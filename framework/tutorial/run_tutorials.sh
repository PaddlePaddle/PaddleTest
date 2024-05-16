#/bin/bash
set +x
CODE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/" && pwd )"
TUTORIALS_EXIT_CODE=0

function run_npu_tutorials(){
    echo "=======Npu Tutorials Test:======="
    failed_cases=''
    cd ${CODE_ROOT}/npu
    for file in $(find ./ -name "test*");do
        echo "=======Tutorials: ${file}======="
        python ${file}
        if [ $? -ne 0 ]; then
            TUTORIALS_EXIT_CODE=8
            if [[ "${failed_cases}" == "" ]];then
                failed_cases="${file}"
            else
                failed_cases="${failed_cases}
${file}"
            fi
        fi
    done
    if [[ "${TUTORIALS_EXIT_CODE}" != "0" ]];then
        echo "=======The Following Tutorials Tests Failed======="
        echo "${failed_cases}"
        echo "--------------------------------------------------"
        echo "The github address of the case is:"
        echo "https://github.com/PaddlePaddle/PaddleTest/blob/develop/framework/tutorial/npu"
        echo "=================================================="
    fi
    exit ${TUTORIALS_EXIT_CODE}
}

function run_tutorials(){
    platform=$1
    echo "=======${platform} Tutorials Test:======="
    failed_cases=''
    cd ${CODE_ROOT}/${platform}
    for file in $(find ./ -name "test*");do
        echo "=======Tutorials: ${file}======="
        python ${file}
        if [ $? -ne 0 ]; then
            TUTORIALS_EXIT_CODE=8
            if [[ "${failed_cases}" == "" ]];then
                failed_cases="${file}"
            else
                failed_cases="${failed_cases}
${file}"
            fi
        fi
    done
    if [[ "${TUTORIALS_EXIT_CODE}" != "0" ]];then
        echo "=======The Following Tutorials Tests Failed======="
        echo "${failed_cases}"
        echo "--------------------------------------------------"
        echo "The github address of the case is:"
        echo "https://github.com/PaddlePaddle/PaddleTest/blob/develop/framework/tutorial/${platform}"
        echo "=================================================="
    fi
    exit ${TUTORIALS_EXIT_CODE}
}

function print_usage(){
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}${SCRIPT_NAME}${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}run_npu_tutorials${NONE}: run npu tutorials\n
    1. Run Npu Tutorials: \n
        'bash run_tutorials.sh run_npu_tutorials' \n
    "
}

function main() {
    local CMD=$1
    local args=("$@")
    case $CMD in
        run_npu_tutorials)
            run_tutorials npu
            ;;
        run_xpu_tutorials)
            run_tutorials xpu
            ;;
        run_dcu_tutorials)
            run_tutorials dcu
            ;;
        run_mlu_tutorials)
            run_tutorials mlu
            ;;
        *)
            print_usage
            ;;
        esac
}

main $@
