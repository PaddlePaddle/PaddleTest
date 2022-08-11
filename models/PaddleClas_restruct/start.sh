cd task
rm -rf models_list_cls_test_ADD
echo Priority_version
echo ${Priority_version}
echo "before ${Priority_version}"
priority_list=(${Priority_version//,/ })
for priority_tmp in ${priority_list[@]}
do
    echo ${priority_tmp}
    if [[ ${priority_tmp} =~ "-" ]];then
        yaml_tmp="ppcls/configs"
        array=(${priority_tmp//-/ })
        for var in ${array[@]}
        do
            export yaml_tmp=${yaml_tmp}/${var}
        done
        export yaml_tmp="${yaml_tmp}.yaml"
        echo ${yaml_tmp} >> models_list_cls_test_ADD
    elif [[ ${priority_tmp} =~ "ImageNet" ]];then
        cat models_list_cls_test_all |grep ImageNet > models_list_cls_test_ImageNet
    else #针对 P0 1 2 3 4
        source creat_scripts.sh ${priority_tmp}
    fi
done
if [[ -f "models_list_cls_test_ADD" ]];then
    source creat_scripts.sh "ADD"
fi
if [[ -f "models_list_cls_test_ImageNet" ]];then
    source creat_scripts.sh "ImageNet"
fi
cd ..
