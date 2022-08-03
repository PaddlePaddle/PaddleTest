rm -rf models_list_cls_test_ADD
echo ${Priority_version}
priority_list=(${Priority_version//,/ })
export Priority_version=" "
for priority_tmp in ${priority_list[@]}
do
    echo ${priority_tmp}
    if [[ ${priority_tmp} =~ "/" ]];then #针对输入多个模型，输入yaml信息
        echo 
        echo ${priority_tmp} >> models_list_cls_test_ADD
        array=(${priority_tmp//\// })
        export model_name=${array[2]} #进行字符串拼接
        for var in ${array[@]:3}
        do
            array2=(${var//'.yaml'/ })
            export model_name=${model_name}_${array2[0]}
        done
        export Priority_version="${Priority_version},${model_name}"  #考虑逗号怎么处理
    else #针对 P0 1 2 3 4
        export Priority_version="${Priority_version},${priority_tmp}"  #考虑逗号怎么处理
        source creat_scripts.sh ${priority_tmp}
    fi
done
export Priority_version=${Priority_version// ,/}
if [[ -f "models_list_cls_test_ADD" ]];then
    source creat_scripts.sh "ADD"
fi
echo ${Priority_version}