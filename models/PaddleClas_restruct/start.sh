cd task
rm -rf models_list_cls_test_ADD
echo "before ${Priority_version}"
priority_list=(${Priority_version//,/ })
for priority_tmp in ${priority_list[@]}
do
    if [[ ${priority_tmp} =~ "-" ]];then
        yaml_tmp="ppcls/configs"
        array=(${priority_tmp//-/ })
        for var in ${array[@]}
        do
            export yaml_tmp=${yaml_tmp}/${var}
        done
        export yaml_tmp="${yaml_tmp}.yaml"
        ehco ${yaml_tmp} >>models_list_cls_test_ADD
    else #针对 P0 1 2 3 4
        source creat_scripts.sh ${priority_tmp}
    fi
done
if [[ -f "models_list_cls_test_ADD" ]];then
    source creat_scripts.sh "ADD"
fi
cd ..
