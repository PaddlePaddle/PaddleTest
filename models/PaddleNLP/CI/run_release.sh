#!/usr/bin/env bash

export P0case_list=()
export P0case_time=0
export all_P0case_time=0
declare -A all_P0case_dic
get_diff_TO_P0case(){
if [[ ${Testcase} == 'all' ]];then
    P0case_list=(waybill_ie msra_ner glue bert skep bigbird electra gpt ernie-1.0 xlnet ofa  squad tinybert lexical_analysis seq2seq \
     word_embedding ernie-ctm distilbert stacl transformer simbert ernie-doc transformer-xl pointer_summarizer question_matching ernie-csc \
    nptag ernie-m clue taskflow transformers)
else
    P0case_list=${Testcase}
fi
}
get_diff_TO_P0case
    echo -e "\033[35m =======CI Check P0case========= \033[0m"
    echo -e "\033[35m ---- P0case_list length: ${#P0case_list[*]}, cases: ${P0case_list[*]} \033[0m"
    set +e
    echo -e "\033[35m ---- start run P0case  \033[0m"
    case_num=1
    for p0case in ${P0case_list[*]};do
        echo -e "\033[35m ---- running P0case $case_num/${#P0case_list[*]}: ${p0case} \033[0m"
        bash ci_case.sh ${p0case}
        let case_num++
    done
    echo -e "\033[35m ---- end run P0case  \033[0m"
cd ${nlp_dir}/
cp -r /ssd1/paddlenlp/bos/* ./
tar -zcvf logs.tar logs/
mkdir upload && mv logs.tar upload
python upload.py upload 'paddle-qa/paddlenlp'
cd logs
FF=`ls *_FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    exit 1
else
    exit 0
fi
