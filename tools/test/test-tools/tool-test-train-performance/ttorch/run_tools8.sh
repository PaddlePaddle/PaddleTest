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

ROOT_PATH=`pwd`

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

cd models/NLP/gpt/
cp ../../../scripts/NLP/gpt/* ./
cp ${ROOT_PATH}/analysis.py ./

sh preData.sh

if [ $1 = "gpt" ];then
    bash run_benchmark.sh sp 8 fp32 200 >log.tmp 2>&1
    flag=$?
    if [ $flag != 0 ];then
	echo -n "trian error"
	exit 1
    fi
    if [ $cards = "1" ];then
        cat nlp_gpt_sp_bs8_fp32_1_speed
    else
	cat nlp_gpt_sp_bs8_fp32_8_speed
    fi
else
    echo -n "model_name error"
    exit 2
fi
