#!/bin/bash
######## how to use ################
#
#
# card=`bash ./get_empty_card.sh`
# export CUDA_VISIBLE_DEVICES=$card
# **ONLY SUPPORT SINGLE CARD**
#
####################################

script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../paddle_log
$script_dir/../paddle_log


cards=`lspci | grep -i nvidia | wc -l`
for((i=0;i<${cards};i++));
do
stat=`nvidia-smi -a -i $i | grep "Process ID"`
if  [ ! -n "$stat" ];then
    sleep 15

    # double check whether gpu card is empty
    stat=`nvidia-smi -a -i $i | grep "Process ID"`
    if  [ ! -n "$stat" ];then
        echo $i
        exit 0
    fi
fi
done
# if no empty card, exit 7
exit 7
