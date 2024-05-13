#/bin/bash
CODE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/" && pwd )"
EXIT_CODE=0

function run_npu_tutorials(){
    cd ${CODE_ROOT}/npu
    for file in $(find ./ -name "test*");do
        echo "=======Tutorials: ${file}======="
        python ${file}
        if [ $? -ne 0 ]; then
            EXIT_CODE=8
        fi
    done
    exit ${EXIT_CODE}
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
            run_npu_tutorials
            ;;
        *)
            print_usage
            ;;
        esac
}

main $@