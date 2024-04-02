####################################
export P0case_list=()
export P0case_time=0
export all_P0case_time=0
declare -A all_P0case_dic
all_P0case_dic=(["waybill_ie"]=3 ["msra_ner"]=15 ["glue"]=2 ["bert"]=2 ["skep"]=10 ["bigbird"]=2 ["electra"]=2  ["gpt"]=2 ["ernie-1.0"]=2 ["xlnet"]=2 \
 ["ofa"]=2   ["squad"]=20 ["tinybert"]=5 ["lexical_analysis"]=5 ["seq2seq"]=5 ["pretrained_models"]=10 ["word_embedding"]=5 \
  ["ernie-ctm"]=5 ["distilbert"]=5  ["stacl"]=5 ["transformer"]=5 ["pet"]=5 ["simbert"]=5 ["ernie-doc"]=20 ["transformer-xl"]=5 \
  ["pointer_summarizer"]=5 ["question_matching"]=5 ["ernie-csc"]=5 ["nptag"]=5 ["ernie-m"]=5 ["clue"]=5)
get_diff_TO_P0case(){
for key in $(echo ${!all_P0case_dic[*]});do
    all_P0case_time=`expr ${all_P0case_time} + ${all_P0case_dic[$key]}`
done
P0case_list=(waybill_ie msra_ner glue bert skep bigbird electra gpt ernie-1.0 xlnet ofa  squad tinybert lexical_analysis seq2seq \
pretrained_models word_embedding ernie-ctm distilbert stacl transformer pet simbert ernie-doc transformer-xl pointer_summarizer question_matching ernie-csc \
nptag ernie-m clue)
P0case_time=${all_P0case_time}
}
set -e
get_diff_TO_P0case
echo -e "\033[35m ---- P0case_list length: ${#P0case_list[*]}, cases: ${P0case_list[*]} \033[0m"
echo -e "\033[35m ---- P0case_time: $P0case_time min \033[0m"
set +e
####################################
echo -e "\033[35m ---- start run P0case  \033[0m"
case_num=1
for p0case in ${P0case_list[*]};do
    echo -e "\033[35m ---- running P0case $case_num/${#P0case_list[*]}: ${p0case} \033[0m"
    echo ${p0case}
    bash run_nlp_all_case.sh ${p0case} 'gpu' 'linux' ${cudaid1} ${cudaid2} 'CI'
    let case_num++
done
echo -e "\033[35m ---- end run P0case  \033[0m"
