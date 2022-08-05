
find ppcls/configs/ImageNet/ -name '*.yaml' -exec ls -l {} \;| awk '{print $NF;}'| grep -v 'eval' \
    | grep -v 'kunlun' |grep -v 'ResNeXt101_32x48d_wsl' |grep -v 'ResNeSt101' > models_list
    #ResNeXt101_32x48d_wsl ResNeSt101 OOM
find ppcls/configs/Cartoonface/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list
find ppcls/configs/Logo/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' >> models_list
find ppcls/configs/Products/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/Vehicle/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/slim/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/GeneralRecognition/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}' \
    |grep -v 'Gallery2FC_PPLCNet_x2_5' >> models_list
find ppcls/configs/DeepHash/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/PULC/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/metric_learning/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
find ppcls/configs/reid/ -name '*.yaml' -exec ls -l {} \; | awk '{print $NF;}'  >> models_list
#317个





if [[ ${model_flag} =~ "pr" ]];then
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep ImageNet|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep Cartoonface|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep Logo|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep Products|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep Vehicle|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep slim|awk -F 'b/' '{print$2}'|tee -a  models_list_diff
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep GeneralRecognition|awk -F 'b/' '{print$2}' |tee -a  models_list_diff
    git diff $(git log --pretty=oneline |grep "Merge pull request"|head -1|awk '{print $1}') HEAD --diff-filter=AMR \
        | grep diff|grep yaml|grep configs|grep DeepHash|awk -F 'b/' '{print$2}' |tee -a  models_list_diff
    shuf -n 3 models_list_diff >> models_list #防止diff yaml文件过多导致pr时间过长
