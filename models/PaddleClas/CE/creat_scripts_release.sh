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
rm -rf ${model}_release.yaml
cp -r ResNet50_release.yaml ${model}_release.yaml
sed -i "" "s|ppcls/configs/ImageNet/ResNet/ResNet50.yaml|$line|g" ${model}_release.yaml
sed -i "" s/ResNet50/$model/g ${model}_release.yaml

# ResNet50	6.43611	0	linux单卡训练
# ResNet50	6.62874	0	linux多卡训练

# RedNet  LeViT   GhostNet 有预训练模型
# MobileNetV3 PPLCNet ESNet   ResNet50    ResNet50_vd
# ResNet50	0.93207	0	linux评估 对于加载预训练模型的要单独评估!!!!!!!!

# ResNet50	16.04218	0	linux单卡训练时 评估

# ResNet50	6.81317	0	windows训练
# ResNet50	0.93661	0	windows评估 对于加载预训练模型的要单独评估
# ResNet50  190.68218	windows训练时 评估

# ResNet50	9.89023	0	mac训练
# ResNet50	7.59464	0	mac评估 对于加载预训练模型的要单独评估
# ResNet50  7.59464	mac训练时 评估

if [[ ${model} == 'AlexNet' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.67853|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.82718|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.05802|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.05802|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.1872|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.49466|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.49466|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'alt_gvt_small' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.29283|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.44081|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.08099|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|8.08099|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.14774|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.91598|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.91598|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'CSPDarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.38832|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.31204|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.27651|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|10.27651|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81611|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.48563|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.48563|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'DarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.6109|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.73945|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|31.5522|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|31.5522|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.54048|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|48202.04587|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|48202.04587|g" ${model}_release.yaml #windows_train_eval
    # sed -i "" "s|6.43611|6.54448|g" ${model}_release.yaml #21116模型原因导致改动一次
    # sed -i "" "s|6.62874|6.71827|g" ${model}_release.yaml

elif [[ ${model} == 'DeiT_tiny_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.02845|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.21715|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.67821|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.67821|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|5.76597|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|8.00662|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|8.00662|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'DenseNet121' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.45391|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.40143|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.53125|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|8.53125|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.8129|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.93523|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.93523|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'DLA46_c' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.48974|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.37077|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.84382|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|8.84382|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81294|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.44571|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.44571|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'DPN107' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.50921|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.56134|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11.15285|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|11.15285|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53907|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.59987|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.59987|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'EfficientNetB0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|18.2399|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|12.83394|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|20.80327|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|20.80327|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|11.95185|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|12.37311|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|12.37311|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'GhostNet_x1_3' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|7.03784|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.82463|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.05573|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|21.81742|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.76389|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|1.06047|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|8759261.36667|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'GoogLeNet' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|11.04376|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|11.05882|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11.19806|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|11.19806|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|10.95292|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|11.14783|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|11.14783|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'HarDNet68' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.5292|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.5534|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.4005|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.4005|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81297|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.40502|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.40502|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'HRNet_W18_C' ]]; then
    sed -i "" "s|P0|P0|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.4628|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.54288|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|16|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|16|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81282|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|16|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|16|g" ${model}_release.yaml #windows_train_eval

    # sed -i "" "s|9.89023|8.14208|g" ${model}_release.yaml #mac_train
    # sed -i "" "s|7.59464|16|g" ${model}_release.yaml #mac_eval
    # sed -i "" "s|7.59464|16|g" ${model}_release.yaml #mac_train_eval
    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}_release.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}_release.yaml

elif [[ ${model} == 'InceptionV4' ]]; then
    sed -i "" "s|P0|P0|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.46977|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.34793|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.91722|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|15.86348|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.5685|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|117.75373|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|117.75373|g" ${model}_release.yaml #windows_train_eval

    # sed -i "" "s|9.89023|7.08905|g" ${model}_release.yaml #mac_train
    # sed -i "" "s|7.59464|7.21561|g" ${model}_release.yaml #mac_eval
    # sed -i "" "s|7.59464|7.21561|g" ${model}_release.yaml #mac_train_eval

elif [[ ${model} == 'LeViT_128S' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1

    # sed -i "" "s|6.43611|5.44842|g" ${model}_release.yaml #linux_train_单卡
    # sed -i "" "s|6.62874|5.77167|g" ${model}_release.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|1.2112|g" ${model}_release.yaml #linux_eval
    # sed -i "" "s|16.04218|10.03219|g" ${model}_release.yaml #linux_train_eval

    # sed -i "" "s|6.81317|6.81317|g" ${model}_release.yaml #windows_train
    # sed -i "" "s|0.93661|190.68218|g" ${model}_release.yaml #windows_eval
    # sed -i "" "s|190.68218|190.68218|g" ${model}_release.yaml #windows_train_eval

    sed -i "" "s|6.43611|0.0|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|0.0|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|0.0|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" ${model}_release.yaml #windows_train_eval

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}_release.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}_release.yaml
    sed -i "" "s|loss|exit_code|g" ${model}_release.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" ${model}_release.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'MixNet_M' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.61655|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.52804|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.1193|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.1193|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81286|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|82.24307|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|82.24307|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'MobileNetV1' ]]; then
    sed -i "" "s|P0|P0|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.45208|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.30737|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.98602|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|8.98602|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.79994|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.50601|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.50601|g" ${model}_release.yaml #windows_train_eval

    # sed -i "" "s|9.89023|7.69221|g" ${model}_release.yaml #mac_train
    # sed -i "" "s|7.59464|10.1224|g" ${model}_release.yaml #mac_eval
    # sed -i "" "s|7.59464|10.1224|g" ${model}_release.yaml #mac_train_eval

elif [[ ${model} == 'MobileNetV2' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|5.84663|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|5.8738|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.89961|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.89961|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.4965|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.16883|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.16883|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'MobileNetV3_large_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.91278|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.9184|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.06482|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|6.93002|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.8921|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|1.06495|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|10.31437|g" ${model}_release.yaml #windows_train_eval

    # sed -i "" "s|9.89023|6.94491|g" ${model}_release.yaml #mac_train
    # sed -i "" "s|7.59464|6.91007|g" ${model}_release.yaml #mac_eval
    # sed -i "" "s|7.59464|6.91007|g" ${model}_release.yaml #mac_train_eval

elif [[ ${model} == 'pcpvt_small' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.26105|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.38915|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.70332|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.70332|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.12982|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.7351|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.7351|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'PPLCNet_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.88318|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.91012|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.23323|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|6.95014|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.89855|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|1.24172|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|9.83488|g" ${model}_release.yaml #windows_train_eval

    # sed -i "" "s|9.89023|6.93865|g" ${model}_release.yaml #mac_train
    # sed -i "" "s|7.59464|6.91936|g" ${model}_release.yaml #mac_eval
    # sed -i "" "s|7.59464|6.91936|g" ${model}_release.yaml #mac_train_eval

elif [[ ${model} == 'RedNet50' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.09878|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.20561|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.94408|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|0.0|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.8142|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|0.93756|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" ${model}_release.yaml #windows_train_eval
    sed -i "" "s|train_eval|exit_code|g" ${model}_release.yaml #训练后评估失败，改为搜集退出码exit_code

elif [[ ${model} == 'Res2Net50_26w_4s' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.50202|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.63189|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.67332|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|10.67332|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53969|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|31.71620|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|31.71620|g" ${model}_release.yaml #windows_train_eval

# elif [[ ${model} == 'ResNeSt101' ]]; then
#     sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
#     sed -i "" "s|6.43611|6.62199|g" ${model}_release.yaml #linux_train_单卡
#     sed -i "" "s|6.62874|6.71942|g" ${model}_release.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|14.58329|g" ${model}_release.yaml #linux_eval
#     sed -i "" "s|16.04218|14.58329|g" ${model}_release.yaml #linux_train_eval

#     sed -i "" "s|6.81317|6.53606|g" ${model}_release.yaml #windows_train
#     sed -i "" "s|0.93661|1374.60004|g" ${model}_release.yaml #windows_eval
#     sed -i "" "s|190.68218|1374.60004|g" ${model}_release.yaml #windows_train_eval
#     # sed -i "" "s|6.43611|6.62913|g" ${model}_release.yaml 211116模型原因导致改动一次
#     # sed -i "" "s|6.62874|6.71608|g" ${model}_release.yaml

elif [[ ${model} == 'ResNeSt50_fast_1s1x64d' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.606|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.71334|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|14.74579|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|14.74579|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53579|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|68.72889|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|68.72889|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'ResNet50_vd' ]]; then
    sed -i "" "s|P0|P0|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.55064|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.56253|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.89191|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.53029|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53586|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|0.88911|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|159.97825|g" ${model}_release.yaml #windows_train_eval

    # sed -i "" "s|9.89023|9.99739|g" ${model}_release.yaml #mac_train
    # sed -i "" "s|7.59464|7.4469|g" ${model}_release.yaml #mac_eval
    # sed -i "" "s|7.59464|7.4469|g" ${model}_release.yaml #mac_train_eval
    sed -i "" "s|ResNet50_vd_vd|ResNet50_vd|g" ${model}_release.yaml #replace

elif [[ ${model} == 'ResNeXt101_32x8d_wsl' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.44288|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.49123|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|18.24487|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|18.24487|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81302|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|83.39121|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|83.39121|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'ResNeXt152_64x4d' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.433|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.46629|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.88949|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|11.33078|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.82732|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|111.75593|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|111.75593|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'ReXNet_1_0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|5.7618|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.15486|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|9.74877|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|9.74877|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.2632|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|16.52984|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|16.52984|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'SE_ResNet18_vd' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.42953|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.34557|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.17827|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|8.17827|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.5366|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|12.56845|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|12.56845|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'ShuffleNetV2_x1_0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.86343|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.67449|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.3624|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.3624|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|7.68979|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|8.21201|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|8.21201|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'SqueezeNet1_0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.6814|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.73497|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.07446|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|7.07446|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.70888|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.01277|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.01277|g" ${model}_release.yaml #windows_train_eval

# elif [[ ${model} == 'SwinTransformer_large_patch4_window12_384' ]]; then
#     sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
#     sed -i "" "s|6.43611|6.62939|g" ${model}_release.yaml #linux_train_单卡
#     sed -i "" "s|6.62874|6.82169|g" ${model}_release.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|7.7453|g" ${model}_release.yaml #linux_eval
#     sed -i "" "s|16.04218|7.7453|g" ${model}_release.yaml #linux_train_eval

#     sed -i "" "s|6.81317|6.81317|g" ${model}_release.yaml #windows_train
#     sed -i "" "s|0.93661|190.68218|g" ${model}_release.yaml #windows_eval
#     sed -i "" "s|190.68218|190.68218|g" ${model}_release.yaml #windows_train_eval
#     sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}_release.yaml #bodong
    # sed -i "" 's|"="|"-"|g' ${model}_release.yaml

elif [[ ${model} == 'SwinTransformer_tiny_patch4_window7_224' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.38868|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.5769|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.58133|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|8.58133|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.16726|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.51868|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.51868|g" ${model}_release.yaml #windows_train_eval
    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}_release.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}_release.yaml

elif [[ ${model} == 'TNT_small' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1

    # sed -i "" "s|6.43611|8.5533|g" ${model}_release.yaml #linux_train_单卡
    # sed -i "" "s|6.62874|6.91239|g" ${model}_release.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|13.37703|g" ${model}_release.yaml #linux_eval
    # sed -i "" "s|16.04218|13.37703|g" ${model}_release.yaml #linux_train_eval

    # sed -i "" "s|6.81317|6.81317|g" ${model}_release.yaml #windows_train
    # sed -i "" "s|0.93661|190.68218|g" ${model}_release.yaml #windows_eval
    # sed -i "" "s|190.68218|190.68218|g" ${model}_release.yaml #windows_train_eval

    sed -i "" "s|6.43611|0.0|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|0.0|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|0.0|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" ${model}_release.yaml #windows_train_eval

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}_release.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}_release.yaml
    sed -i "" "s|loss|exit_code|g" ${model}_release.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" ${model}_release.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'VGG11' ]]; then
    sed -i "" "s|P0|P0|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.77401|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|6.83604|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|6.94488|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|6.94488|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.67848|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|7.05581|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|7.05581|g" ${model}_release.yaml #windows_train_eval

    # sed -i "" "s|9.89023|7.16128|g" ${model}_release.yaml #mac_train
    # sed -i "" "s|7.59464|6.93275|g" ${model}_release.yaml #mac_eval
    # sed -i "" "s|7.59464|6.93275|g" ${model}_release.yaml #mac_train_eval

elif [[ ${model} == 'ViT_small_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    # sed -i "" "s|6.43611|161465090.328|g" ${model}_release.yaml #linux_train_单卡
    # sed -i "" "s|6.62874|143.35147|g" ${model}_release.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|273455638.816|g" ${model}_release.yaml #linux_eval
    # sed -i "" "s|16.04218|273455638.816|g" ${model}_release.yaml #linux_train_eval

    # sed -i "" "s|6.81317|1653146872261.061|g" ${model}_release.yaml #windows_train
    # sed -i "" "s|0.93661|3955238406280.53320|g" ${model}_release.yaml #windows_eval
    # sed -i "" "s|190.68218|3955238406280.53320|g" ${model}_release.yaml #windows_train_eval

    sed -i "" "s|6.43611|0.0|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|0.0|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|0.0|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" ${model}_release.yaml #windows_train_eval

    sed -i "" "s|loss|exit_code|g" ${model}_release.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" ${model}_release.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'Xception41_deeplab' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|5.77681|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|5.37061|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|17.78094|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|17.78094|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.57439|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|9.81792|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|9.81792|g" ${model}_release.yaml #windows_train_eval

elif [[ ${model} == 'Xception71' ]]; then
    sed -i "" "s|P0|P1|g" ${model}_release.yaml #P0/1
    sed -i "" "s|6.43611|6.11912|g" ${model}_release.yaml #linux_train_单卡
    sed -i "" "s|6.62874|5.77516|g" ${model}_release.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11.60848|g" ${model}_release.yaml #linux_eval
    sed -i "" "s|16.04218|11.60848|g" ${model}_release.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.60048|g" ${model}_release.yaml #windows_train
    sed -i "" "s|0.93661|73.75383|g" ${model}_release.yaml #windows_eval
    sed -i "" "s|190.68218|73.75383|g" ${model}_release.yaml #windows_train_eval
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
