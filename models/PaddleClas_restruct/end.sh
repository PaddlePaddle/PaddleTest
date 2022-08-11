cd ${Project_path} #确定下执行路径
echo "end.sh"
echo $PWD

if [[ ${get_data_way} == "ln_way" ]];then
    if [[ ${Data_path} == "" ]];then
        echo " you must set Data_path first "
    fi
    echo "do not need rm Data_path"
else
    # 增加一下判断，别直接cd到空
    if [[ -d dataset ]];then
        echo $(ls dataset|head -n 2)
        echo " have dataset"
        rm -rf dataset
    fi

    infer_tar=(`echo *_infer.tar`)
    if [[ ${infer_tar} == "*_infer.tar" ]];then
        echo " do not have infer_tar"
    else
        rm -rf *_infer.tar
    fi

    infer_file=(`echo *_infer`)
    if [[ ${infer_file} == "*_infer" ]];then
        echo " do not have infer_file"
    else
        rm -rf *_infer*
    fi

    pretrained_pdparams=(`echo *_pretrained.pdparams`)
    if [[ ${pretrained_pdparams} == "_pretrained.pdparams" ]];then
        echo " do not have pretrained_pdparams"
    else
        rm -rf *_pretrained*
    fi
fi

cd ../../
#回收数据，避免产生过多缓存
