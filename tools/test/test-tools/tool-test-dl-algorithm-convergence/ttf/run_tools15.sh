model_name=${1:-"mnist"}
cards=${2:-"1"}

if [ $cards = "1" ];then
     export CUDA_VISIBLE_DEVICES=0
elif [ $cards = "8" ];then
     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    echo -n "cards error"
    exit -1
fi

if [ ${model_name} = "mnist" ];then
    sh prepare.sh >log.prepare 2>&1
    sh run_mnist.sh
    flag=$?
    if [ $flag == 1 ];then
	echo -n "trian error"
	exit 1
    elif [ $flag == 2 ];then
	echo -n "not convergent"
	exit 2
    fi
    cat models/official/vision/image_classification/log.conv
else
    echo -n "model_name error"
    exit 3
fi
