find ../PaddleGAN/configs/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'\
    | grep -v 'wav2lip' \
    | grep -v 'edvr_l_blur_wo_tsa' \
    | grep -v 'edvr_l_blur_w_tsa' \
    | grep -v 'mprnet_deblurring' \
    | grep -v 'msvsr_l_reds' \
    > PaddleGAN_ALL_list
    #OOM

#P优先级删除all中没有的
cat PaddleGAN_P0_list |while read line; do if [[ ! `grep -c "${line}" PaddleGAN_ALL_list` -ne '0' ]] ;then
    index=`grep -n ${line} PaddleGAN_P0_list | awk -F ":" '{print $1}'`
    sed -i "" "${index}d" PaddleGAN_P0_list ; fi; done  #注意linux 不要 ""
    # echo ${index} ; fi; done
cat PaddleGAN_P1_list |while read line; do if [[ ! `grep -c "${line}" PaddleGAN_ALL_list` -ne '0' ]] ;then
    index=`grep -n ${line} PaddleGAN_P1_list | awk -F ":" '{print $1}'`
    sed -i "" "${index}d" PaddleGAN_P1_list ; fi; done  #注意linux 不要 ""
    # echo ${index} ; fi; done


#P优先级插入all中新增的
cat PaddleGAN_ALL_list |while read line; do if [[ ! `grep -c "${line}" PaddleGAN_P0_list` -ne '0' ]] \
    && [[ ! `grep -c "${line}" PaddleGAN_P1_list` -ne '0' ]]  ;then echo $line; fi; done


# cat models_list | sort | uniq > models_list_run_tmp  #去重复

# if [[ ${model_flag} =~ "pr" ]];then
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep ImageNet|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep Cartoonface|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
