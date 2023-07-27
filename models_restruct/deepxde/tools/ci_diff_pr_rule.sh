
find ppcls/configs/ImageNet/ -name '*.yaml' -exec ls -l {} \;| awk '{print $NF;}'| grep -v 'eval' \
    |grep -v 'ResNeXt101_32x48d_wsl' |grep -v 'ResNeSt101' |grep -v 'ConvNeXt' \
    |grep -v 'RedNet152' \
    |grep -v 'RedNet101' \
    |grep -v 'PVT_V2_B4' \
    |grep -v 'DeiT_base_patch16_384' \
    |grep -v 'EfficientNetB4' \
    |grep -v 'DeiT_base_patch16_224' \
    |grep -v 'EfficientNetB5' \
    |grep -v 'DeiT_base_distilled_patch16_224' \
    |grep -v 'VisionTransformer-ViT_large_patch16_224' \
    |grep -v 'VisionTransformer-ViT_large_patch32_384' \
    |grep -v 'SwinTransformer_base_patch4_window12_384' \
    |grep -v 'SwinTransformer_base_patch4_window7_224' \
    |grep -v 'SwinTransformer_large_patch4_window12_384' \
    |grep -v 'SwinTransformer_large_patch4_window7_224' \
    |grep -v 'EfficientNetB6' \
    |grep -v 'DeiT_base_distilled_patch16_384' \
    |grep -v 'EfficientNetB7' \
    |grep -v 'ViT_large_patch16_384' \
    |grep -v 'PVT_V2_B5' \
    |grep -v 'mv3_large_x1_0_distill_mv3_small_x1_0' \
    |grep -v 'ResNeXt101_32x32d_wsl' \
    |grep -v 'amp_' \
    > models_list
    #OOM 支持原 bs/3 向上取整
    #amp 单独处理
find ppcls/configs/Cartoonface/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list
find ppcls/configs/Logo/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list
find ppcls/configs/Products/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/Vehicle/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/slim/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/GeneralRecognition/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' \
    |grep -v 'Gallery2FC_PPLCNet_x2_5' >> models_list
    #Gallery2FC_PPLCNet_x2_5 格式不正确
find ppcls/configs/GeneralRecognitionV2/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/DeepHash/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/PULC/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' |grep -wv 'search.yaml' \
    |grep -v 'Res2Net200_vd_26w_4s' |grep -v 'SwinTransformer_tiny_patch4_window7_224' >> models_list
    #search.yaml格式不正确
    # Res2Net200_vd_26w_4s、SwinTransformer_tiny_patch4_window7_224时间过长重要性低
# find ppcls/configs/metric_learning/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
#221031暂时去掉，数据太大总下载不下来
find ppcls/configs/reid/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
# MAC执行后 会有// 区分linux只有/  要进行替换//
#282个模型总量
#11个类型
#新增了1个类型TODO添加
# ppcls^configs^ImageNet^Distillation^mv3_large_x1_0_distill_mv3_small_x1_0.yaml
#P优先级删除all中没有的
cat models_list_cls_test_P0 |while read line; do if [[ ! `grep -c "${line}" models_list_cls_test_all` -ne '0' ]] ;then
    index=`grep -n ${line} models_list_cls_test_P0 | awk -F ":" '{print $1}'`
    sed -i "" "${index}d" models_list_cls_test_P0 ; fi; done  #注意linux 不要 ""
    # echo ${index} ; fi; done
cat models_list_cls_test_P1 |while read line; do if [[ ! `grep -c "${line}" models_list_cls_test_all` -ne '0' ]] ;then
    index=`grep -n ${line} models_list_cls_test_P1 | awk -F ":" '{print $1}'`
    sed -i "" "${index}d" models_list_cls_test_P1 ; fi; done  #注意linux 不要 ""
    # echo ${index} ; fi; done
cat models_list_cls_test_P2 |while read line; do if [[ ! `grep -c "${line}" models_list_cls_test_all` -ne '0' ]] ;then
    index=`grep -n ${line} models_list_cls_test_P2 | awk -F ":" '{print $1}'`
    sed -i "" "${index}d" models_list_cls_test_P2 ; fi; done  #注意linux 不要 ""
    # echo ${index} ; fi; done
cat models_list_cls_test_P3 |while read line; do if [[ ! `grep -c "${line}" models_list_cls_test_all` -ne '0' ]] ;then
    index=`grep -n ${line} models_list_cls_test_P3 | awk -F ":" '{print $1}'`
    sed -i "" "${index}d" models_list_cls_test_P3 ; fi; done  #注意linux 不要 ""
    # echo ${index} ; fi; done


#P优先级插入all中新增的
cat models_list_cls_test_all |while read line; do if [[ ! `grep -c "${line}" models_list_cls_test_P0` -ne '0' ]] \
    && [[ ! `grep -c "${line}" models_list_cls_test_P1` -ne '0' ]] \
    && [[ ! `grep -c "${line}" models_list_cls_test_P2` -ne '0' ]] \
    && [[ ! `grep -c "${line}" models_list_cls_test_P3` -ne '0' ]]  ;then echo $line; fi; done


# cat models_list | sort | uniq > models_list_run_tmp  #去重复

# ppcls/configs/ImageNet/ConvNeXt/ConvNeXt_tiny.yaml  P2
# 暂时剔除，hang  意思内存泄露打回

# CSWinTransformer_base_384
# maybe hang  先放回去

# if [[ ${model_flag} =~ "pr" ]];then
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep ImageNet|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep Cartoonface|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep Logo|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep Products|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep Vehicle|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep slim|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep GeneralRecognition|awk -F 'b/' '{print$2}' |tee -a  models_list_diff
#     git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
#         | grep diff|grep yaml|grep configs|grep DeepHash|awk -F 'b/' '{print$2}' |tee -a  models_list_diff
#     shuf -n 3 models_list_diff >> models_list #防止diff yaml文件过多导致pr时间过长
