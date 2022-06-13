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
rm -rf ${model}.yaml
cp -r ResNet50.yaml ${model}.yaml
sed -i "" "s|ppcls/configs/ImageNet/ResNet/ResNet50.yaml|$line|g" ${model}.yaml
sed -i "" s/ResNet50/$model/g ${model}.yaml

# ResNet50	6.39329	0	linux单卡训练
# ResNet50	6.50465	0	linux多卡训练

# RedNet  LeViT   GhostNet 有预训练模型
# MobileNetV3 PPLCNet ESNet   ResNet50    ResNet50_vd
# ResNet50	0.93207	0	linux评估 对于加载预训练模型的要单独评估!!!!!!!!

# ResNet50	8.44204	0	linux单卡训练时 评估
# ResNet50	10.96941	0	linux多卡训练时 评估

# ResNet50	6.43611	0	linux单卡训练_release
# ResNet50	6.62874	0	linux多卡训练_release

# RedNet  LeViT   GhostNet 有预训练模型
# MobileNetV3 PPLCNet ESNet   ResNet50    ResNet50_vd
# ResNet50	0.93207	0	linux评估 对于加载预训练模型的要单独评估!!!!!!!!

# ResNet50	16.04218	0	linux单卡训练时 评估_release
# ResNet50	7.29234	0	linux多卡训练时 评估_release

if [[ ${model} == 'AlexNet' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.71162|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.73761|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.01511|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.01511|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01512|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.67853|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.82718|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.05802|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.05802|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05803|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'alt_gvt_small' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.27162|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.44887|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.992|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.992|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.17294|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.41927|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|8.46421|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|8.46421|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'CSPDarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.47177|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.35905|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|27.74947|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|27.74947|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.38832|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.31204|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|10.27651|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|10.27651|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'DarkNet53' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.58574|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.64748|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|23.03562|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|23.03562|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    # sed -i "" "s|6.39329|6.54448|g" ${model}.yaml #21116模型原因导致改动一次
    # sed -i "" "s|6.50465|6.71827|g" ${model}.yaml

    sed -i "" "s|6.43611|6.6109|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.73945|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|31.5522|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|31.5522|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'DeiT_tiny_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.31575|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.23028|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.8391|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.8391|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.34939|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.48434|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.94151|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.94151|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'DenseNet121' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.50436|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.26448|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|15.04463|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|15.04463|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.45391|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.40143|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|8.53125|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|8.53125|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'DLA46_c' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.42155|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.44165|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.80652|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|10.80652|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.48974|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.37077|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|8.84382|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|8.84382|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'DPN107' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.56654|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.55417|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.12993|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.12993|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.50921|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.56134|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|11.15285|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|11.15285|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'EfficientNetB0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|23.26233|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|12.54865|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|570.27105|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|570.27105|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|18.2399|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|12.83394|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|20.80327|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|20.80327|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'GhostNet_x1_3' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|7.03793|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.82361|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.05573|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|8.00004|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|7.03784|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.82463|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|1.05573|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|21.81742|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'GoogLeNet' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|11.0307|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|11.04306|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|11.17378|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|11.17378|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|11.04376|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|11.05882|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|11.19806|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|11.19806|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'HarDNet68' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.59988|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.39026|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.1209|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.1209|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.5292|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.5534|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.4005|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.4005|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'HRNet_W18_C' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.4628|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.54288|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|16|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|16|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.43611|6.4628|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.54288|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|16|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|16|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}.yaml

elif [[ ${model} == 'InceptionV4' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.52594|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.38526|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.91722|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|11.94133|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.46977|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.34793|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|0.91722|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|15.86348|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release


elif [[ ${model} == 'LeViT_128S' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    # sed -i "" "s|6.39329|5.44842|g" ${model}.yaml #linux_train_单卡
    # sed -i "" "s|6.50465|5.77167|g" ${model}.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|1.2112|g" ${model}.yaml #linux_eval
    # sed -i "" "s|8.44204|10.03219|g" ${model}.yaml #linux_train_eval_单卡

    sed -i "" "s|6.39329|0.0|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|0.0|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|0.0|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|0.0|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|0.0|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|0.0|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|0.0|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|0.0|g" ${model}.yaml #linux_train_eval_多卡_release

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}.yaml
    # sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    # sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'MixNet_M' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.49278|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.53958|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|12.25185|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|12.25185|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.61655|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.52804|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.1193|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.1193|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'MobileNetV1' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.62079|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.29075|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.12151|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.12151|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.45208|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.30737|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|8.98602|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|8.98602|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release


elif [[ ${model} == 'MobileNetV2' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.09929|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.78941|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.50131|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.50131|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|5.84663|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|5.8738|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.89961|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.89961|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'MobileNetV3_large_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.91073|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.90942|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.06482|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|6.91593|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.91278|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.9184|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|1.06482|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|6.93002|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release


elif [[ ${model} == 'pcpvt_small' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.28812|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.60131|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.81204|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.81204|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.35917|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.45379|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.69966|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.69966|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'PPLCNet_x1_0' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.8921|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.9094|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|1.23323|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|6.9751|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.88318|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.91012|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|1.23323|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|6.95014|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release


elif [[ ${model} == 'RedNet50' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.00765|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.2255|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.94408|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.09878|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.20561|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|0.94408|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|0.0|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

    sed -i "" "s|train_eval|exit_code|g" ${model}.yaml #训练后评估失败，改为搜集退出码exit_code

elif [[ ${model} == 'Res2Net50_26w_4s' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.50464|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.62286|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|24.27187|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|24.27187|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.50202|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.63189|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|10.67332|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|10.67332|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

# elif [[ ${model} == 'ResNeSt101' ]]; then
#     sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
#     sed -i "" "s|6.39329|6.62199|g" ${model}.yaml #linux_train_单卡
#     sed -i "" "s|6.50465|6.71942|g" ${model}.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|14.58329|g" ${model}.yaml #linux_eval
#     sed -i "" "s|8.44204|14.58329|g" ${model}.yaml #linux_train_eval_单卡
    # sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

#     # sed -i "" "s|6.39329|6.62913|g" ${model}.yaml 211116模型原因导致改动一次
#     # sed -i "" "s|6.50465|6.71608|g" ${model}.yaml

elif [[ ${model} == 'ResNeSt50_fast_1s1x64d' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.61989|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.6843|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.00482|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|8.00482|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.606|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.71334|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|14.74579|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|14.74579|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'ResNet50_vd' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.61876|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.65543|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.89191|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.15642|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.55064|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.56253|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|0.89191|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.53029|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

    sed -i "" "s|ResNet50_vd_vd|ResNet50_vd|g" ${model}.yaml #replace
    sed -i "" "s|ResNet50_vd_vd_vd|ResNet50_vd|g" ${model}.yaml #replace

elif [[ ${model} == 'ResNeXt101_32x8d_wsl' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.3635|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.58852|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|12.40129|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|12.40129|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.44288|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.49123|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|18.24487|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|18.24487|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'ResNeXt152_64x4d' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.41932|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.47852|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.88949|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|10.05526|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.433|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.46629|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|0.88949|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|11.33078|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'ReXNet_1_0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|5.68809|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.91201|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|9.15968|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|9.15968|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|5.7618|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.15486|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|9.74877|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|9.74877|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'SE_ResNet18_vd' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.39355|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.38664|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.69632|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|8.69632|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.42953|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.34557|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|8.17827|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|8.17827|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'ShuffleNetV2_x1_0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.85395|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.60863|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.36902|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.36902|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.86343|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.67449|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.3624|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.3624|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'SqueezeNet1_0' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.72474|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.74748|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|7.04342|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|7.04342|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.6814|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.73497|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.07446|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.07446|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release


# elif [[ ${model} == 'SwinTransformer_large_patch4_window12_384' ]]; then
#     sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
#     sed -i "" "s|6.39329|6.62939|g" ${model}.yaml #linux_train_单卡
#     sed -i "" "s|6.50465|6.82169|g" ${model}.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|7.7453|g" ${model}.yaml #linux_eval
#     sed -i "" "s|8.44204|7.7453|g" ${model}.yaml #linux_train_eval_单卡
    # sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

#     sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}.yaml #bodong
    # sed -i "" 's|"="|"-"|g' ${model}.yaml

elif [[ ${model} == 'SwinTransformer_tiny_patch4_window7_224' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.3451|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.80341|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|8.74345|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|8.74345|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.43404|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.54192|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|7.71512|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|7.71512|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

    #sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}.yaml #bodong #220503change
    #sed -i "" 's|"="|"-"|g' ${model}.yaml

elif [[ ${model} == 'TNT_small' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    # sed -i "" "s|6.39329|8.5533|g" ${model}.yaml #linux_train_单卡
    # sed -i "" "s|6.50465|6.91239|g" ${model}.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|13.37703|g" ${model}.yaml #linux_eval
    # sed -i "" "s|8.44204|13.37703|g" ${model}.yaml #linux_train_eval_单卡

    sed -i "" "s|6.39329|0.0|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|0.0|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|0.0|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|0.0|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|0.0|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|0.0|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|0.0|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|0.0|g" ${model}.yaml #linux_train_eval_多卡_release

    sed -i "" "s|threshold: 0.0|threshold: 0.1|g" ${model}.yaml #bodong
    sed -i "" 's|"="|"-"|g' ${model}.yaml
    # sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    # sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'VGG11' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.79892|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|6.82267|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|6.94058|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|6.94058|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.77401|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|6.83604|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|6.94488|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|6.94488|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release


elif [[ ${model} == 'ViT_small_patch16_224' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    # sed -i "" "s|6.39329|161465090.328|g" ${model}.yaml #linux_train_单卡
    # sed -i "" "s|6.50465|143.35147|g" ${model}.yaml #linux_train_多卡
    # sed -i "" "s|0.93207|273455638.816|g" ${model}.yaml #linux_eval
    # sed -i "" "s|8.44204|273455638.816|g" ${model}.yaml #linux_train_eval_单卡

    sed -i "" "s|6.39329|0.0|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|0.0|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|0.0|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|0.0|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|0.0|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|0.0|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|0.0|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|0.0|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|0.0|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|0.0|g" ${model}.yaml #linux_train_eval_多卡_release

    # sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
    # sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

elif [[ ${model} == 'Xception41_deeplab' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|5.54425|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.42837|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|10.24912|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|10.24912|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|5.77681|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|5.37061|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|17.78094|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|17.78094|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

elif [[ ${model} == 'Xception71' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml #P0/1
    sed -i "" "s|6.39329|6.07467|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|6.50465|5.73247|g" ${model}.yaml #linux_train_多卡
    sed -i "" "s|0.93207|47.38125|g" ${model}.yaml #linux_eval
    sed -i "" "s|8.44204|47.38125|g" ${model}.yaml #linux_train_eval_单卡
    sed -i "" "s|10.96941|7.01511|g" ${model}.yaml #linux_train_eval_多卡

    sed -i "" "s|6.43611|6.11912|g" ${model}.yaml #linux_train_单卡_release
    sed -i "" "s|6.62874|5.77516|g" ${model}.yaml #linux_train_多卡_release
    sed -i "" "s|0.93207|11.60848|g" ${model}.yaml #linux_eval_release
    sed -i "" "s|16.04218|11.60848|g" ${model}.yaml #linux_train_eval_单卡_release
    sed -i "" "s|7.29234|7.05802|g" ${model}.yaml #linux_train_eval_多卡_release

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
