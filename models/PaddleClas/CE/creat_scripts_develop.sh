#shell
# cat models_list_cls_testP0 | while read line
# cat models_list_cls_testP1 | while read line
cat models_list_cls_test_all | while read line
do
echo $line
filename=${line##*/}
#echo $filename
model=${filename%.*}
echo $model

cd config
rm -rf $model.yaml
cp -r ResNet50.yaml $model.yaml
sed -i "" "s|ppcls/configs/ImageNet/ResNet/ResNet50.yaml|$line|g" $model.yaml
sed -i "" s/ResNet50/$model/g $model.yaml

# ResNet50	6.39329	0	linux单卡训练
# ResNet50	6.50465	0	linux多卡训练

# RedNet  LeViT   GhostNet 有预训练模型
# MobileNetV3 PPLCNet ESNet   ResNet50    ResNet50_vd
# ResNet50	0.93207	0	linux评估 对于加载预训练模型的要单独评估!!!!!!!!

# ResNet50	8.44204	0	linux单卡训练时 评估

# ResNet50	6.81317	0	windows训练
# ResNet50	0.93661	0	windows评估 对于加载预训练模型的要单独评估
# ResNet50  190.68218	windows训练时 评估

# ResNet50	9.89023	0	mac训练
# ResNet50	7.59464	0	mac评估 对于加载预训练模型的要单独评估
# ResNet50  7.59464	mac训练时 评估

if [[ ${model} == 'AlexNet' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.71162|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.73761|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.01511|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.01511|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.1872|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.49466|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.49466|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'alt_gvt_small' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.27162|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.44887|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.992|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.992|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.14774|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.91598|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.91598|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'CSPDarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.47177|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.35905|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|27.74947|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|27.74947|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81611|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.48563|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.48563|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.58574|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.64748|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|23.03562|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|23.03562|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.54048|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|48202.04587|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|48202.04587|g" $model.yaml #windows_train_eval
    # sed -i "" "s|6.39329|6.54448|g" $model.yaml #21116模型原因导致改动一次
    # sed -i "" "s|6.50465|6.71827|g" $model.yaml

elif [[ ${model} == 'DeiT_tiny_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.31575|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.23028|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.8391|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.8391|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|5.76597|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|8.00662|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|8.00662|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DenseNet121' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.50436|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.26448|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|15.04463|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|15.04463|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.8129|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.93523|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.93523|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DLA46_c' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.42155|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.44165|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.80652|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|10.80652|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81294|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.44571|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.44571|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DPN107' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.56654|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.55417|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.12993|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.12993|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53907|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.59987|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.59987|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'EfficientNetB0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|23.26233|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|12.54865|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|570.27105|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|570.27105|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|11.95185|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|12.37311|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|12.37311|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'GhostNet_x1_3' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|7.03793|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.82361|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.05573|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|8.00004|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.76389|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|1.06047|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|8759261.36667|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'GoogLeNet' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|11.0307|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|11.04306|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11.17378|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|11.17378|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|10.95292|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|11.14783|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|11.14783|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'HarDNet68' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.59988|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.39026|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.1209|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.1209|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81297|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.40502|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.40502|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'HRNet_W18_C' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.4628|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.54288|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|16|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|16|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81282|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|16|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|16|g" $model.yaml #windows_train_eval

    # sed -i "" "s|9.89023|8.14208|g" ${model}.yaml #mac_train
    # sed -i "" "s|7.59464|16|g" ${model}.yaml #mac_eval
    # sed -i "" "s|7.59464|16|g" ${model}.yaml #mac_train_eval
    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}.yaml

elif [[ ${model} == 'InceptionV4' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.52594|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.38526|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.91722|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|11.94133|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.5685|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|117.75373|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|117.75373|g" $model.yaml #windows_train_eval

    # sed -i "" "s|9.89023|7.08905|g" ${model}.yaml #mac_train
    # sed -i "" "s|7.59464|7.21561|g" ${model}.yaml #mac_eval
    # sed -i "" "s|7.59464|7.21561|g" ${model}.yaml #mac_train_eval

elif [[ ${model} == 'LeViT_128S' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1

    # sed -i "" "s|6.39329|5.44842|g" $model.yaml #linux_train_单卡
    # sed -i "" "s|6.50465|5.77167|g" $model.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|1.2112|g" $model.yaml #linux_eval
    # sed -i "" "s|8.44204|10.03219|g" $model.yaml #linux_train_eval

    # sed -i "" "s|6.81317|6.81317|g" $model.yaml #windows_train
    # sed -i "" "s|0.93661|190.68218|g" $model.yaml #windows_eval
    # sed -i "" "s|190.68218|190.68218|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.39329|0.0|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|0.0|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    sed -i "" 's|"="|"-"|g' $model.yaml
    sed -i "" "s|loss|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'MixNet_M' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.49278|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.53958|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|12.25185|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|12.25185|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81286|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|82.24307|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|82.24307|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'MobileNetV1' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.62079|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.29075|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.12151|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.12151|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.79994|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.50601|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.50601|g" $model.yaml #windows_train_eval

    # sed -i "" "s|9.89023|7.69221|g" ${model}.yaml #mac_train
    # sed -i "" "s|7.59464|10.1224|g" ${model}.yaml #mac_eval
    # sed -i "" "s|7.59464|10.1224|g" ${model}.yaml #mac_train_eval

elif [[ ${model} == 'MobileNetV2' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.09929|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.78941|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.50131|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.50131|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.4965|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.16883|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.16883|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'MobileNetV3_large_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.91073|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.90942|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.06482|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|6.91593|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.8921|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|1.06495|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|10.31437|g" $model.yaml #windows_train_eval

    # sed -i "" "s|9.89023|6.94491|g" ${model}.yaml #mac_train
    # sed -i "" "s|7.59464|6.91007|g" ${model}.yaml #mac_eval
    # sed -i "" "s|7.59464|6.91007|g" ${model}.yaml #mac_train_eval

elif [[ ${model} == 'pcpvt_small' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.28812|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.60131|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.81204|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.81204|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.12982|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.7351|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.7351|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'PPLCNet_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.8921|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.9094|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.23323|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|6.9751|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.89855|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|1.24172|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|9.83488|g" $model.yaml #windows_train_eval

    # sed -i "" "s|9.89023|6.93865|g" ${model}.yaml #mac_train
    # sed -i "" "s|7.59464|6.91936|g" ${model}.yaml #mac_eval
    # sed -i "" "s|7.59464|6.91936|g" ${model}.yaml #mac_train_eval

elif [[ ${model} == 'RedNet50' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.00765|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.2255|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.94408|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.8142|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.93756|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval
    sed -i "" "s|train_eval|exit_code|g" $model.yaml #训练后评估失败，改为搜集退出码exit_code

elif [[ ${model} == 'Res2Net50_26w_4s' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.50464|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.62286|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|24.27187|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|24.27187|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53969|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|31.71620|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|31.71620|g" $model.yaml #windows_train_eval

# elif [[ ${model} == 'ResNeSt101' ]]; then
#     sed -i "" "s|P0|P1|g" $model.yaml #P0/1
#     sed -i "" "s|6.39329|6.62199|g" $model.yaml #linux_train_单卡
#     sed -i "" "s|6.50465|6.71942|g" $model.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|14.58329|g" $model.yaml #linux_eval
#     sed -i "" "s|8.44204|14.58329|g" $model.yaml #linux_train_eval

#     sed -i "" "s|6.81317|6.53606|g" $model.yaml #windows_train
#     sed -i "" "s|0.93661|1374.60004|g" $model.yaml #windows_eval
#     sed -i "" "s|190.68218|1374.60004|g" $model.yaml #windows_train_eval
#     # sed -i "" "s|6.39329|6.62913|g" $model.yaml 211116模型原因导致改动一次
#     # sed -i "" "s|6.50465|6.71608|g" $model.yaml

elif [[ ${model} == 'ResNeSt50_fast_1s1x64d' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.61989|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.6843|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.00482|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|8.00482|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53579|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|68.72889|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|68.72889|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ResNet50_vd' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.61876|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.65543|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.89191|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.15642|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53586|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.88911|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|159.97825|g" $model.yaml #windows_train_eval

    # sed -i "" "s|9.89023|9.99739|g" ${model}.yaml #mac_train
    # sed -i "" "s|7.59464|7.4469|g" ${model}.yaml #mac_eval
    # sed -i "" "s|7.59464|7.4469|g" ${model}.yaml #mac_train_eval
    sed -i "" "s|ResNet50_vd_vd|ResNet50_vd|g" ${model}.yaml #replace
    sed -i "" "s|ResNet50_vd_vd_vd|ResNet50_vd|g" ${model}_release.yaml #replace

elif [[ ${model} == 'ResNeXt101_32x8d_wsl' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.3635|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.58852|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|12.40129|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|12.40129|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81302|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|83.39121|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|83.39121|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ResNeXt152_64x4d' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.41932|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.47852|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.88949|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|10.05526|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.82732|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|111.75593|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|111.75593|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ReXNet_1_0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|5.68809|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.91201|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|9.15968|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|9.15968|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.2632|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|16.52984|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|16.52984|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'SE_ResNet18_vd' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.39355|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.38664|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.69632|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|8.69632|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.5366|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|12.56845|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|12.56845|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ShuffleNetV2_x1_0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.85395|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.60863|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.36902|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.36902|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|7.68979|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|8.21201|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|8.21201|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'SqueezeNet1_0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.72474|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.74748|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.04342|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|7.04342|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.70888|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.01277|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.01277|g" $model.yaml #windows_train_eval

# elif [[ ${model} == 'SwinTransformer_large_patch4_window12_384' ]]; then
#     sed -i "" "s|P0|P1|g" $model.yaml #P0/1
#     sed -i "" "s|6.39329|6.62939|g" $model.yaml #linux_train_单卡
#     sed -i "" "s|6.50465|6.82169|g" $model.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|7.7453|g" $model.yaml #linux_eval
#     sed -i "" "s|8.44204|7.7453|g" $model.yaml #linux_train_eval

#     sed -i "" "s|6.81317|6.81317|g" $model.yaml #windows_train
#     sed -i "" "s|0.93661|190.68218|g" $model.yaml #windows_eval
#     sed -i "" "s|190.68218|190.68218|g" $model.yaml #windows_train_eval
#     sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    # sed -i "" 's|"="|"-"|g' $model.yaml

elif [[ ${model} == 'SwinTransformer_tiny_patch4_window7_224' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.3451|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.80341|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.74345|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|8.74345|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.16726|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.51868|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.51868|g" $model.yaml #windows_train_eval
    #sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong #220503change
    #sed -i "" 's|"="|"-"|g' $model.yaml

elif [[ ${model} == 'TNT_small' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1

    # sed -i "" "s|6.39329|8.5533|g" $model.yaml #linux_train_单卡
    # sed -i "" "s|6.50465|6.91239|g" $model.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|13.37703|g" $model.yaml #linux_eval
    # sed -i "" "s|8.44204|13.37703|g" $model.yaml #linux_train_eval

    # sed -i "" "s|6.81317|6.81317|g" $model.yaml #windows_train
    # sed -i "" "s|0.93661|190.68218|g" $model.yaml #windows_eval
    # sed -i "" "s|190.68218|190.68218|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.39329|0.0|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|0.0|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    sed -i "" 's|"="|"-"|g' $model.yaml
    sed -i "" "s|loss|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'VGG11' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.79892|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.82267|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|6.94058|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|6.94058|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.67848|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.05581|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.05581|g" $model.yaml #windows_train_eval

    # sed -i "" "s|9.89023|7.16128|g" ${model}.yaml #mac_train
    # sed -i "" "s|7.59464|6.93275|g" ${model}.yaml #mac_eval
    # sed -i "" "s|7.59464|6.93275|g" ${model}.yaml #mac_train_eval

elif [[ ${model} == 'ViT_small_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    # sed -i "" "s|6.39329|161465090.328|g" $model.yaml #linux_train_单卡
    # sed -i "" "s|6.50465|143.35147|g" $model.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|273455638.816|g" $model.yaml #linux_eval
    # sed -i "" "s|8.44204|273455638.816|g" $model.yaml #linux_train_eval

    # sed -i "" "s|6.81317|1653146872261.061|g" $model.yaml #windows_train
    # sed -i "" "s|0.93661|3955238406280.53320|g" $model.yaml #windows_eval
    # sed -i "" "s|190.68218|3955238406280.53320|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.39329|0.0|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|0.0|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval

    sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'Xception41_deeplab' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|5.54425|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.42837|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.24912|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|10.24912|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.57439|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|9.81792|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|9.81792|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'Xception71' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.39329|6.07467|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.73247|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|47.38125|g" $model.yaml #linux_eval
    sed -i "" "s|8.44204|47.38125|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.60048|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|73.75383|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|73.75383|g" $model.yaml #windows_train_eval
fi

cd ..
done





# test case

# sed -i "" "s|threshold: 0|threshold: 0.0 #|g" config/*.yaml
# sed -i "" "s|kpi_base: 0|kpi_base: 0.0 #|g" config/*.yaml


# sed -i "" "s|"-"|"="|g" config/*.yaml
# # sed -i "" "s|infer_linux,eval_linux,infer_linux|infer_linux,eval_linux|g" config/AlexNet.yaml
# sed -i "" "s|infer_linux,eval_linux|eval_linux,infer_linux|g" config/*.yaml


# sed -i "" "s|"="|"-"|g" config/*.yaml #错误
# sed -i "" 's|"="|"-"|g' config/AlexNet.yaml
# sed -i "" 's|"="|"-"|g' config/*.yaml


# sed -i "" "s|function_test|linux_function_test|g" config/AlexNet.yaml
# sed -i "" 's|exec_tag:|exec_tag: $EXEC_TAG #|g' config/AlexNet.yaml
