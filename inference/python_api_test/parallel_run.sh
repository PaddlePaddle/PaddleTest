set -m
home=$PWD
EXIT_CODE=0
run_dirs=(test_class_model test_det_model test_ocr_model test_nlp_model)
if [[ -z $1 ]];then
    card_number=1
else
    card_number=$1
fi
case_number=${#run_dirs[@]}
#TODO后续动态获取显卡梳理
#card_number=$(nvidia-smi -L | wc -l)
step=$card_number
EXIT_CODE=0;
function caught_error() {
    for job in `jobs -p`; do
        echo "PID => ${job}"
        if ! wait ${job} ; then
            echo "At least one test failed with exit code => $?";
            let EXIT_CODE=EXIT_CODE+1;
        fi
    done
}
#trap 'caught_error' CHLD
for((i=0;i<case_number;i+=step))
do
    trap 'caught_error' CHLD
    for((j=0;j<step;j++))
    do
        if [[ -z "${run_dirs[i+j]}" ]];then
            break
        else
           echo "${run_dirs[i+j]}"
           cd $home/${run_dirs[i+j]}/ && bash run.sh $j 2>&1 &
        fi
    done
    wait
done
#展示结果并设置返回值
cd $home
echo "show me the result"
#TODO 展示错误case
if [[ $EXIT_CODE -gt 0 ]];
then
     exit 8
else
     exit 0
fi
