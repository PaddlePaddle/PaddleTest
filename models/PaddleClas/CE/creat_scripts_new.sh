#shell
#只有在模型库增加新的模型，或者改变P0/1结构，执行生成yaml程序

# RedNet  LeViT   GhostNet 有预训练模型
# MobileNetV3 PPLCNet ESNet   ResNet50    ResNet50_vd

#以resnet50为基准，自动从report中获取如下基准值

# ResNet50	6.39329	0	linux单卡训练
# ResNet50	6.50465	0	linux多卡训练
# ResNet50	0.93207	0	linux评估 对于加载预训练模型的要单独评估!!!!!!!!
# ResNet50	8.44204	0	linux单卡训练时 评估
# ResNet50	10.96941	0	linux多卡训练时 评估

# ResNet50	6.43611	0	linux单卡训练_release
# ResNet50	6.62874	0	linux多卡训练_release
# ResNet50	0.93207	0	linux评估 对于加载预训练模型的要单独评估!!!!!!!!
# ResNet50	16.04218	0	linux单卡训练时 评估_release
# ResNet50	7.29234	0	linux多卡训练时 评估_release

# python analysis_html.py #搜集value值  develop 和 release分开统计
#     如何解析？？？现在能把报告转成array
#     从网页解析判断合规，先自己生成后解析   done

# models_list_cls_testP0 弄成循环可配置化  done

# 如果resnet50 release 和 develop一致，其它的不一致怎么办
#     基准值不能只局限于resnet50，也应该可配置化，增加判断条件如果是作为基准值的模型不复制（要用全等号）  done


export base_model=ResNet50
export base_priority=P0
priority_all='P0 P1' # P0 P1 #还可以控制单独生成某一个yaml models_list_cls_test${某一个或几个模型}
branch='develop release'  # develop release  #顺序不能反
# read -p "Press enter to continue"  #卡一下

echo base_model
echo $base_model
for priority_tmp in $priority_all
do
    cat models_list_cls_test_${priority_tmp} | while read line
    do
        echo $line
        filename=${line##*/}
        #echo $filename
        model=${filename%.*}
        echo $model

        if [[ ${base_model} = ${model} ]]; then
            # echo "#####"
            continue
        else
            # echo "@@@@@"
            cd config
            rm -rf ${model}.yaml
            cp -r ${base_model}.yaml ${model}.yaml
            sed -i "" "s|ppcls/configs/ImageNet/ResNet/${base_model}.yaml|$line|g" ${model}.yaml #待优化，去掉ResNet
            sed -i "" s/${base_model}/$model/g ${model}.yaml

            #记录一些特殊规则
            if [[ ${model} == 'HRNet_W18_C' ]]; then
                sed -i "" "s|threshold: 0.0|threshold: 0.2 #|g" ${model}.yaml #bodong
                sed -i "" 's|"="|"-"|g' ${model}.yaml
            elif [[ ${model} == 'LeViT_128S' ]]; then
                sed -i "" "s|threshold: 0.0|threshold: 0.2 #|g" ${model}.yaml #bodong
                sed -i "" 's|"="|"-"|g' ${model}.yaml
                # sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
                # sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
            elif [[ ${model} == 'RedNet50' ]]; then
                sed -i "" "s|train_eval|exit_code|g" ${model}.yaml #训练后评估失败，改为搜集退出码exit_code
            elif [[ ${model} == 'ResNet50_vd' ]]; then
                sed -i "" "s|ResNet50_vd_vd|ResNet50_vd|g" ${model}.yaml #replace
                sed -i "" "s|ResNet50_vd_vd_vd|ResNet50_vd|g" ${model}.yaml #replace
            # elif [[ ${model} == 'TNT_small' ]]; then
            #     sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}.yaml #bodong
            #     sed -i "" 's|"="|"-"|g' ${model}.yaml
                # sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
                # sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
                #暂时监控linux通过改学习率不出nan
            # elif [[ ${model} == 'ViT_small_patch16_224' ]]; then
            #     sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
                # sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
            fi
            for branch_tmp in $branch
            do

                # echo branch_tmp
                # echo $branch_tmp
                # echo $branch
                # 在这里还要判断当前模型在report中是否存在，不存在的话就不执行
                if [[ ! `grep -c "${model}" ../clas_${branch_tmp}` -ne '0' ]] ;then
                    echo "new model :${model}"
                else
                    # echo priority_tmp
                    # echo ${base_priority}
                    # echo ${priority_tmp}
                    # echo $base_model
                    # echo $branch_tmpbranch
                    # grep "${base_model}" ../clas_${branch_tmp}
                    # read -p "Press enter to continue"  #卡一下

                    sed -i "" "s|"${base_priority}"|"${priority_tmp}"|g" ${model}.yaml #P0/1 #不加\$会报（正常的） sed: first RE may not be empty 加了值不会变
                    arr_base=($(echo `grep -w "${base_model}" ../clas_${branch_tmp}` | awk 'BEGIN{FS=",";OFS=" "} {print $1,$2,$3,$4,$5,$6,$7,$8}'))
                    arr_target=($(echo `grep -w "${model}" ../clas_${branch_tmp}` | awk 'BEGIN{FS=",";OFS=" "} {print $1,$2,$3,$4,$5,$6,$7,$8}'))
                    # echo arr_base
                    # echo ${arr_base[*]}
                    # echo ${arr_target[*]}
                    num_lisrt='1 2 3 4 5 6 7 8' #一共有8个值需要改变
                    for num_lisrt_tmp in $num_lisrt
                        do
                        # echo ${arr_base[${num_lisrt_tmp}]}
                        # echo ${arr_target[${num_lisrt_tmp}]}
                        sed -i "" "1,/"${arr_base[${num_lisrt_tmp}]}"/s/"${arr_base[${num_lisrt_tmp}]}"/"${arr_target[${num_lisrt_tmp}]}"/" ${model}.yaml
                        #mac命令只替换第一个，linux有所区别需要注意
                        # sed -i "" "s|"${arr_base[${num_lisrt_tmp}]}"|"${arr_target[${num_lisrt_tmp}]}"|g" ${model}.yaml #linux_train_单卡

                        done
                fi
            done

            cd ..
        fi

    done

done
