#!/usr/bin/env bash
####################################
# get diff case
export P0case_list=()
export APIcase_list=()
declare -A Normal_dic
declare -A all_P0case_dic
all_P0case_dic=(["waybill_ie"]=3 ["msra_ner"]=15 ["glue"]=2 ["bert"]=2 ["skep"]=10 ["bigbird"]=2 ["electra"]=2  ["gpt"]=2 ["ernie-1.0"]=2 ["xlnet"]=2 \
 ["ofa"]=2 ["albert"]=2   ["SQuAD"]=20 ["tinybert"]=5 ["lexical_analysis"]=5 ["seq2seq"]=5 ["pretrained_models"]=10 ["word_embedding"]=5 \
  ["ernie-ctm"]=5 ["distilbert"]=5  ["stacl"]=5 ["transformer"]=5 ["pet"]=5 ["simbert"]=5 ["ernie-doc"]=20 ["transformer-xl"]=5 \
  ["pointer_summarizer"]=5 ["question_matching"]=5 ["ernie-csc"]=5 ["nptag"]=5 ["ernie-m"]=5 ["taskflow"]=5 ["clue"]=5 ["textcnn"]=5)
for line in `cat model_list.txt`;do
    all_example_dict[${#all_example_dict[*]}]=$line
done
get_diff_TO_P0case(){
for file_name in `git diff --numstat origin |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    echo "file_name:"${file_name}, "dir1:"${dir1}, "dir2:"${dir2},"dir3:"${dir3},".xx:" ${file_name##*.}
    if [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
        continue
    elif [[ ${dir1} =~ "paddlenlp" ]];then # API 升级
        if [[ ${!all_P0case_dic[*]} =~ ${dir2} ]];then
            P0case_list[${#P0case_list[*]}]=${dir2}
        elif [[ ${dir2} =~ "transformers" ]];then
            if [[ ${dir3} == "ernie_m" ]];then
                P0case_list[${#P0case_list[*]}]=ernie-m
            elif [[ ${dir3} == "ernie_doc" ]];then
                P0case_list[${#P0case_list[*]}]=ernie-doc
            elif [[ ${dir3} == "ernie_ctm" ]];then
                P0case_list[${#P0case_list[*]}]=ernie-ctm
            elif [[ ${dir3} == "ernie" ]];then
                P0case_list[${#P0case_list[*]}]=ernie-1.0
            elif [[ ${!all_P0case_dic[*]} =~ ${dir3} ]];then
                P0case_list[${#P0case_list[*]}]=${dir3}
            else
                P0case_list[${#P0case_list[*]}]=bert
                P0case_list[${#P0case_list[*]}]=gpt
                P0case_list[${#P0case_list[*]}]=transformer
            fi
        fi
    elif [[ ${dir1} =~ "examples" ]];then # 模型升级
        if [[ ${!all_P0case_dic[*]} =~ ${dir3} ]];then
            P0case_list[${#P0case_list[*]}]=${dir3}
        elif [[ ${dir3##*.} == "py" ]] && [[ !(${all_example_dict[*]} =~ ${dir2}) ]];then #新增规范模型
            P0case_list[${#P0case_list[*]}]=${dir2}
            Normal_dic[${dir2}]="${dir1}/${dir2}/"
        elif [[ !(${all_example_dict[*]} =~ ${dir3}) ]] ;then
            P0case_list[${#P0case_list[*]}]=${dir3}
            Normal_dic[${dir3}]="${dir1}/${dir2}/${dir3}"
        fi
    elif [[ ${dir1} =~ "model_zoo" ]];then # 模型升级
        if [[ ${!all_P0case_dic[*]} =~ ${dir2} ]];then
            P0case_list[${#P0case_list[*]}]=${dir2}
        elif [[ !(${all_example_dict[*]} =~ ${dir2}) ]];then #新增规范模型
            P0case_list[${#P0case_list[*]}]=${dir2}
            Normal_dic[${dir2}]="${dir1}/${dir2}/"
        fi
    elif [[ ${dir1} =~ "tests" ]];then #新增单测
        if [[ ${dir3##*.} == "py" ]];then
            continue
        elif [[ ${dir2} =~ "transformers" ]] ;then
            APIcase_list[${#APIcase_list[*]}]=${dir3}
        fi
    else
        continue
    fi
done
}
get_diff_TO_P0case
P0case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${P0case_list[*]}))
APIcase_list=($(awk -v RS=' ' '!a[$1]++' <<< ${APIcase_list[*]}))
####################################
if [[ ${#P0case_list[*]} -ne 0 ]] || [[ ${#APIcase_list[*]} -ne 0 ]];then
    ####################################
    # set python env
    case $1 in
    27)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs2/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-2.7.15-ucs2/bin/:${PATH}
    ;;
    35)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
    ;;
    36)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
    ;;
    37)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
    ;;
    38)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
    ;;
    esac
    python -c 'import sys; print(sys.version_info[:])'
    echo "python="$1
    ####################################
    # set paddle env
    set -x
    python -m pip install --ignore-installed --upgrade pip
    python -m pip install -r requirements_ci.txt
    python -m pip install $2;
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
    python -c 'from visualdl import LogWriter'
    ####################################
    # set paddlenlp env
    nlp1_build (){
        echo -e "\033[35m ---- only install paddlenlp \033[0m"
        python -m pip install -U paddlenlp
    }
    nlp2_build (){
        echo -e "\033[35m ---- build and install paddlenlp  \033[0m"
        rm -rf build/
        rm -rf paddlenlp.egg-info/
        rm -rf dist/

        python -m pip install --ignore-installed -r requirements.txt
        python setup.py bdist_wheel
        python -m pip install --ignore-installed  dist/paddlenlp****.whl
    }
    $3
    export NLTK_DATA=/ssd1/paddlenlp/nltk_data/
    pip list
    set +x
    ####################################
    # set logs env
    export nlp_dir=/workspace
    mkdir /workspace/model_logs
    mkdir /workspace/unittest_logs
    mkdir /workspace/coverage_logs
    export log_path=/workspace/model_logs
    ####################################
    # run changed models case
    echo -e "\033[35m =======CI Check P0case========= \033[0m"
    echo -e "\033[35m ---- P0case_list length: ${#P0case_list[*]}, cases: ${P0case_list[*]} \033[0m"
    set +e
    echo -e "\033[35m ---- start run P0case  \033[0m"
    case_num=1
    for p0case in ${P0case_list[*]};do
        echo -e "\033[35m ---- running P0case $case_num/${#P0case_list[*]}: ${p0case} \033[0m"
        if [[ ${!Normal_dic[*]} =~ ${example} ]];then
            bash normal_case.sh ${Normal_dic[${example}]}
            let case_num++
        else
            bash pr_case.sh ${example}
            let case_num++
        fi
    done
    echo -e "\033[35m ---- end run P0case  \033[0m"
    cd ${nlp_dir}/model_logs
    FF=`ls *FAIL*|wc -l`
    EXCODE=0
    if [ "${FF}" -gt "0" ];then
        P0case_EXCODE=1
        EXCODE=2
    else
        P0case_EXCODE=0
    fi
    if [ $P0case_EXCODE -ne 0 ] ; then
        echo -e "\033[31m ---- P0case Failed number: ${FF} \033[0m"
        ls *_FAIL*
    else
        echo -e "\033[32m ---- P0case Success \033[0m"
    fi
    ####################################
    # run unittest
    cd ${nlp_dir}
    echo -e "\033[35m =======CI Check Unittest========= \033[0m"
    echo -e "\033[35m ---- unittest length: ${#APIcase_list[*]}, unittest cases: ${APIcase_list[*]} \033[0m"
    for apicase in ${APIcase_list[*]};do
        pytest tests/transformers/${apicase}/test_*.py  >${nlp_dir}/unittest_logs/${apicase}_unittest.log 2>&1
        # sh run_coverage.sh paddlenlp.transformers.${apicase} >unittest_logs/${apicase}_coverage.log 2>&1
        UT_EXCODE=$? || true
        if [ $UT_EXCODE -ne 0 ] ; then
            mv ${nlp_dir}/unittest_logs/${apicase}_unittest.log ${nlp_dir}/unittest_logs/${apicase}_unittest_FAIL.log
        fi
    done
    cd ${nlp_dir}/unittest_logs
    UF=`ls *FAIL*|wc -l`
    if [ "${UF}" -gt "0" ];then
        UT_EXCODE=1
        EXCODE=3
    else
        UT_EXCODE=0
    fi
    if [ $UT_EXCODE -ne 0 ] ; then
        echo -e "\033[31m ---- Unittest Failed \033[0m"
        ls *_FAIL*
    else
        echo -e "\033[32m ---- Unittest Success \033[0m"
    fi
    ####################################
    # run coverage
    # cd ${nlp_dir}/tests/
    # bash run_coverage.sh
    # Coverage_EXCODE=$? || true
    # mv ./htmlcov ${nlp_dir}/coverage_logs/
    # if [ $Coverage_EXCODE -ne 0 ] ; then
    #     echo -e "\033[31m ---- Coverage Failed \033[0m"
    # else
    #     echo -e "\033[32m ---- Coverage Success \033[0m"
    # fi
    ####################################
else
    echo -e "\033[32m Changed files no in ci case, Skips \033[0m"
    EXCODE=0
fi
exit $EXCODE
