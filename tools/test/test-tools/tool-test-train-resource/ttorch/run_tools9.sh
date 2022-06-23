unset GREP_OPTIONS

model_name=${1:-"gpt"}
cards=${2:-"1"}

if [ $cards = "1" ];then
     export CUDA_VISIBLE_DEVICES=0
elif [ $cards = "8" ];then
     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    echo -n "cards error"
    exit -1
fi

git clone https://github.com/PaddlePaddle/benchmark.git
cd benchmark/OtherFrame/nlp/PyTorch/
/bin/rm -rf apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 70d4a0ba0e9bc740cd9d1982d73c159ed4d76e6c
cd -
cd models/NLP/gpt/
rm -rf Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout e269e200649555069454bf78335466f9658b2044
cd -

sh PrepareEnv.sh

cp ../../../scripts/NLP/gpt/* ./
cp ../../../analysis.py ./

sh preData.sh

if [ $1 = "gpt" ];then
# 统计GPU显存占用
rm -rf  gpu_use.log
gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu,memory.used --format=csv -lms 100 > gpu_use.log 2>&1 &
gpu_memory_pid=$!
# 统计CPU
time=$(date "+%Y-%m-%d %H:%M:%S")
LAST_CPU_INFO=$(cat /proc/stat | grep -w cpu | awk '{print $2,$3,$4,$5,$6,$7,$8}')
LAST_SYS_IDLE=$(echo $LAST_CPU_INFO | awk '{print $4}')
LAST_TOTAL_CPU_T=$(echo $LAST_CPU_INFO | awk '{print $1+$2+$3+$4+$5+$6+$7}')

CURR_PATH=`pwd`
bash run_benchmark.sh sp 8 fp32 200 >log.tmp 2>&1
flag=$?
if [ $flag != 0 ];then
    echo "trian error"
    exit 1
fi
cd ${CURR_PATH}

NEXT_CPU_INFO=$(cat /proc/stat | grep -w cpu | awk '{print $2,$3,$4,$5,$6,$7,$8}')
NEXT_SYS_IDLE=$(echo $NEXT_CPU_INFO | awk '{print $4}')
NEXT_TOTAL_CPU_T=$(echo $NEXT_CPU_INFO | awk '{print $1+$2+$3+$4+$5+$6+$7}')

#系统空闲时间
SYSTEM_IDLE=`echo ${NEXT_SYS_IDLE} ${LAST_SYS_IDLE} | awk '{print $1-$2}'`
#CPU总时间
TOTAL_TIME=`echo ${NEXT_TOTAL_CPU_T} ${LAST_TOTAL_CPU_T} | awk '{print $1-$2}'`
#echo "LAST_SYS_IDLE:" $LAST_SYS_IDLE
#echo "NEXT_SYS_IDLE:" $NEXT_SYS_IDLE
#echo "LAST_TOTAL_CPU_T:" $LAST_TOTAL_CPU_T
#echo "NEXT_TOTAL_CPU_T:" $NEXT_TOTAL_CPU_T
#echo "SYSTEM_IDLE:" $SYSTEM_IDLE
#echo "TOTAL_TIME: " $TOTAL_TIME
if [ $TOTAL_TIME == 0 ];then  # 两次系统的总时间一致,说明CPU的使用的时间计划为0
    AVG_CPU_USE=0
else
     CPU_USAGE=`echo ${SYSTEM_IDLE} ${TOTAL_TIME} | awk '{printf "%.0f", (1-$1/$2)*100}'`
     AVG_CPU_USE=${CPU_USAGE}
fi

# 计算显存占用
kill ${gpu_memory_pid}
MAX_GPU_MEMORY_USE=`awk 'BEGIN {max = 0} {if(NR>1){if ($3 > max) max=$3}} END {print max}' gpu_use.log`
AVG_GPU_USE=`awk '{if(NR>1 && $1 >0){time+=$1;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("%.0f\n" ,avg)}' gpu_use.log`

echo -n {"avg_cpu_util": 0.$AVG_CPU_USE, "max_gpu_memory_usage_mb": $MAX_GPU_MEMORY_USE, "avg_gpu_util": 0.$AVG_GPU_USE}


else
    echo -n "model_name error"
    exit 2
fi
