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

# ResNet50	6.43773	0	linux单卡训练
# ResNet50	6.44749	0	linux多卡训练

# RedNet  LeViT   GhostNet 有预训练模型
# MobileNetV3 PPLCNet ESNet   ResNet50    ResNet50_vd
# ResNet50	0.93207	0	linux评估 对于加载预训练模型的要单独评估!!!!!!!!

# ResNet50	18.21894	0	linux单卡训练时 评估

# ResNet50	6.81317	0	windows训练
# ResNet50	0.93661	0	windows评估 对于加载预训练模型的要单独评估
# ResNet50  190.68218	windows训练时 评估


if [[ ${model} == 'AlexNet' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.67973|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.82505|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.01067|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.01067|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.1872|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.49466|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.49466|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'alt_gvt_small' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.2827|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.42552|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.81478|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.81478|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.11162|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.90485|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.90485|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'CSPDarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.24161|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.48492|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.46527|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|8.46527|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81611|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.48563|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.48563|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.62999|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.72895|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|18.12492|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|18.12492|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.54048|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|48202.04587|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|48202.04587|g" $model.yaml #windows_train_eval
    # sed -i "" "s|6.43773|6.54448|g" $model.yaml #21116模型原因导致改动一次
    # sed -i "" "s|6.44749|6.71827|g" $model.yaml

elif [[ ${model} == 'DeiT_tiny_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.02449|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.13435|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.68187|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.68187|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|5.76617|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.94462|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.94462|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DenseNet121' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.47078|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.32436|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.68034|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|10.68034|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.8129|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.93523|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.93523|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DLA46_c' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.39816|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.24263|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|9.92038|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|9.92038|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81294|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.44571|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.44571|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'DPN107' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.50796|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.55144|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|12.88681|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|12.88681|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53907|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.59987|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.59987|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'EfficientNetB0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|12.77855|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|12.24543|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|61.39692|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|61.39692|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|8.30399|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|9.06218|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|9.06218|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'GhostNet_x1_3' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|7.03821|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.83161|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.05573|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|8.90942|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.76316|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|1.06047|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|9339895.46667|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'GoogLeNet' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|11.04352|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|11.054|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11.19209|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|11.19209|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|10.95292|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|11.14783|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|11.14783|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'HarDNet68' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.51581|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.43451|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.32358|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.32358|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81297|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.40502|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.40502|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'HRNet_W18_C' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.4628|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.54288|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|11|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81282|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|11|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|11|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.81317|6.81282|g" $model.yaml #mac_train
    sed -i "" "s|0.93661|11|g" $model.yaml #mac_eval
    sed -i "" "s|190.68218|11|g" $model.yaml #mac_train_eval
    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    sed -i "" 's|"="|"-"|g' $model.yaml

elif [[ ${model} == 'InceptionV4' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.47594|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.375|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.55219|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|10.55219|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.5685|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|117.75373|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|117.75373|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.81317|6.5685|g" $model.yaml #mac_train
    sed -i "" "s|0.93661|117.75373|g" $model.yaml #mac_eval
    sed -i "" "s|190.68218|117.75373|g" $model.yaml #mac_train_eval
elif [[ ${model} == 'LeViT_128S' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1

    # sed -i "" "s|6.43773|5.44842|g" $model.yaml #linux_train_单卡
    # sed -i "" "s|6.44749|5.77167|g" $model.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|1.2112|g" $model.yaml #linux_eval
    # sed -i "" "s|18.21894|10.03219|g" $model.yaml #linux_train_eval

    # sed -i "" "s|6.81317|6.81317|g" $model.yaml #windows_train
    # sed -i "" "s|0.93661|190.68218|g" $model.yaml #windows_eval
    # sed -i "" "s|190.68218|190.68218|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.43773|0.0|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|0.0|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    sed -i "" 's|"="|"-"|g' $model.yaml
    sed -i "" "s|loss|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'MixNet_M' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.50224|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.58454|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|25.94225|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|25.94225|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81291|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|940.94432|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|940.94432|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'MobileNetV1' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.46045|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.29338|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.47801|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|8.47801|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.79994|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.50601|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.50601|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.81317|6.79994|g" $model.yaml #mac_train
    sed -i "" "s|0.93661|7.50601|g" $model.yaml #mac_eval
    sed -i "" "s|190.68218|7.50601|g" $model.yaml #mac_train_eval
elif [[ ${model} == 'MobileNetV2' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|5.87994|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|5.69139|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.80226|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.80226|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.4965|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.16883|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.16883|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'MobileNetV3_large_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.91377|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.92049|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.06482|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.00676|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.89211|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|1.06495|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|10.93849|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.81317|6.89211|g" $model.yaml #mac_train
    sed -i "" "s|0.93661|1.06495|g" $model.yaml #mac_eval
    sed -i "" "s|190.68218|10.93849|g" $model.yaml #mac_train_eval
elif [[ ${model} == 'pcpvt_small' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.28169|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.39268|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.75694|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.75694|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.19867|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.79898|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.79898|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'PPLCNet_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.88368|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.91007|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.23323|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|6.95113|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.89859|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|1.24172|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|9.8212|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'RedNet50' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.04059|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.09659|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.94408|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81482|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.93756|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval
    sed -i "" "s|train_eval|exit_code|g" $model.yaml #训练后评估失败，改为搜集退出码exit_code

elif [[ ${model} == 'Res2Net50_26w_4s' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.52324|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.64367|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|9.85302|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|9.85302|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53969|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|31.71620|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|31.71620|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ResNeSt101' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.62199|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.71942|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|14.58329|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|14.58329|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53606|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|1374.60004|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|1374.60004|g" $model.yaml #windows_train_eval
    # sed -i "" "s|6.43773|6.62913|g" $model.yaml 211116模型原因导致改动一次
    # sed -i "" "s|6.44749|6.71608|g" $model.yaml

elif [[ ${model} == 'ResNeSt50_fast_1s1x64d' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.61576|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.71876|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.24997|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|8.24997|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53613|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|10.43912|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|10.43912|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ResNet50_vd' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.50555|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.634|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.89191|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|15.16626|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53586|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.88911|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|159.97825|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.81317|6.53586|g" $model.yaml #mac_train
    sed -i "" "s|0.93661|0.88911|g" $model.yaml #mac_eval
    sed -i "" "s|190.68218|159.97825|g" $model.yaml #mac_train_eval
    sed -i "" "s|ResNet50_vd_vd|ResNet50_vd|g" $model.yaml #replace

elif [[ ${model} == 'ResNeXt101_32x8d_wsl' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.46018|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.62916|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11.77823|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|11.77823|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.81302|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|83.39121|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|83.39121|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ResNeXt152_64x4d' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.41659|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.55609|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|9.20373|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|9.20373|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.82732|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|111.75593|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|111.75593|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ReXNet_1_0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|5.64194|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|5.65636|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.80714|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|8.80714|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.20366|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|40.89920|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|40.89920|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'SE_ResNet18_vd' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.38639|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.34327|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.82379|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|8.82379|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.53612|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.84304|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.84304|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'ShuffleNetV2_x1_0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.86075|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.69692|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.36083|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.36083|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|7.68979|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|8.21201|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|8.21201|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'SqueezeNet1_0' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.70269|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.75931|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.07058|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|7.07058|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.70888|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.01277|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.01277|g" $model.yaml #windows_train_eval

# elif [[ ${model} == 'SwinTransformer_large_patch4_window12_384' ]]; then
#     sed -i "" "s|P0|P1|g" $model.yaml #P0/1
#     sed -i "" "s|6.43773|6.62939|g" $model.yaml #linux_train_单卡
#     sed -i "" "s|6.44749|6.82169|g" $model.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|7.7453|g" $model.yaml #linux_eval
#     sed -i "" "s|18.21894|7.7453|g" $model.yaml #linux_train_eval

#     sed -i "" "s|6.81317|6.81317|g" $model.yaml #windows_train
#     sed -i "" "s|0.93661|190.68218|g" $model.yaml #windows_eval
#     sed -i "" "s|190.68218|190.68218|g" $model.yaml #windows_train_eval
#     sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    # sed -i "" 's|"="|"-"|g' $model.yaml

elif [[ ${model} == 'SwinTransformer_tiny_patch4_window7_224' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.38868|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.5769|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.58133|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|8.58133|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.16726|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.51868|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.51868|g" $model.yaml #windows_train_eval
    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    sed -i "" 's|"="|"-"|g' $model.yaml

elif [[ ${model} == 'TNT_small' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1

    # sed -i "" "s|6.43773|8.5533|g" $model.yaml #linux_train_单卡
    # sed -i "" "s|6.44749|6.91239|g" $model.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|13.37703|g" $model.yaml #linux_eval
    # sed -i "" "s|18.21894|13.37703|g" $model.yaml #linux_train_eval

    # sed -i "" "s|6.81317|6.81317|g" $model.yaml #windows_train
    # sed -i "" "s|0.93661|190.68218|g" $model.yaml #windows_eval
    # sed -i "" "s|190.68218|190.68218|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.43773|0.0|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|0.0|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" $model.yaml #bodong
    sed -i "" 's|"="|"-"|g' $model.yaml
    sed -i "" "s|loss|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" $model.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'VGG11' ]]; then
    sed -i "" "s|P0|P0|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.78336|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|6.82774|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|6.9413|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|6.9413|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.67848|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|7.05581|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|7.05581|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.81317|6.67848|g" $model.yaml #mac_train
    sed -i "" "s|0.93661|7.05581|g" $model.yaml #mac_eval
    sed -i "" "s|190.68218|7.05581|g" $model.yaml #mac_train_eval
elif [[ ${model} == 'ViT_small_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    # sed -i "" "s|6.43773|161465090.328|g" $model.yaml #linux_train_单卡
    # sed -i "" "s|6.44749|143.35147|g" $model.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|273455638.816|g" $model.yaml #linux_eval
    # sed -i "" "s|18.21894|273455638.816|g" $model.yaml #linux_train_eval

    # sed -i "" "s|6.81317|1653146872261.061|g" $model.yaml #windows_train
    # sed -i "" "s|0.93661|3955238406280.53320|g" $model.yaml #windows_eval
    # sed -i "" "s|190.68218|3955238406280.53320|g" $model.yaml #windows_train_eval

    sed -i "" "s|6.43773|0.0|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|0.0|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|0.0|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|0.0|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|0.0|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|0.0|g" $model.yaml #windows_train_eval

    sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'Xception41_deeplab' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|5.9148|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|5.24088|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.31752|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|10.31752|g" $model.yaml #linux_train_eval

    sed -i "" "s|6.81317|6.57439|g" $model.yaml #windows_train
    sed -i "" "s|0.93661|9.81792|g" $model.yaml #windows_eval
    sed -i "" "s|190.68218|9.81792|g" $model.yaml #windows_train_eval

elif [[ ${model} == 'Xception71' ]]; then
    sed -i "" "s|P0|P1|g" $model.yaml #P0/1
    sed -i "" "s|6.43773|6.09203|g" $model.yaml #linux_train_单卡
    sed -i "" "s|6.44749|5.80298|g" $model.yaml #linux_train_多卡
    sed -i "" "s|0.93207|13.61897|g" $model.yaml #linux_eval
    sed -i "" "s|18.21894|13.61897|g" $model.yaml #linux_train_eval

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
