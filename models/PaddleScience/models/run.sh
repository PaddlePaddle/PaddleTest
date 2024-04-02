wget https://paddle-qa.bj.bcebos.com/PaddleScience/model/CE_data/CE_GPU_ALL.tar.gz
tar -xzf CE_GPU_ALL.tar.gz


cases=`find . -maxdepth 1 -name "*.py" | sort `
ignore="tool.py"
serial_bug=0
distributed_bug=0
bug=0

export CUDA_VISIBLE_DEVICES=0
echo "===== examples bug list =====" >  result.txt
echo "serial bug list:" >>  result.txt
for file in ${cases}
do
echo serial ${file} test
if [[ ${ignore} =~ ${file##*/} ]]; then
    echo "skip"
else
    python3.7 ${file}
    if [ $? -ne 0 ]; then
        echo ${file} >> result.txt
        bug=`expr ${bug} + 1`
        serial_bug=`expr ${serial_bug} + 1`
    fi
fi
done
echo "serial bugs: "${serial_bug} >> result.txt


export CUDA_VISIBLE_DEVICES=0,1
ignore="tool.py"
echo "distributed bug list:" >>  result.txt
for file in ${cases}
do
echo distributed ${file} test
if [[ ${ignore} =~ ${file##*/} ]]; then
    echo "skip"
else
    python3.7 -m paddle.distributed.launch --devices=0,1  ${file}
    if [ $? -ne 0 ]; then
        echo ${file} >> result.txt
        bug=`expr ${bug} + 1`
        distributed_bug=`expr ${distributed_bug} + 1`
    fi
fi
done
echo "distributed bugs: "${distributed_bug} >> result.txt

echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}
