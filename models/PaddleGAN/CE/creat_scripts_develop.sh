#shell
# cat models_list_gan_testP0 | while read line
# cat models_list_gan_testP1 | while read line
cat models_list_gan_test_all | while read line
do
echo $line
filename=${line##*/}
model=${filename%.*}
echo ${model}

cd config
rm -rf ${model}.yaml
cp -r edvr_m_wo_tsa.yaml ${model}.yaml
sed -i "" "s|config/edvr_m_wo_tsa.yaml|$line|g" ${model}.yaml
sed -i "" s/edvr_m_wo_tsa/${model}/g ${model}.yaml

# edvr_m_wo_tsa	35000	0	linux单卡训练
# edvr_m_wo_tsa	35000	0	linux多卡训练
# edvr_m_wo_tsa	35000	0	linux_eval
# edvr_m_wo_tsa	35000	0	linux_train_eval

# edvr_m_wo_tsa	6.81317	0	windows训练
# edvr_m_wo_tsa	0.93661	0	windows评估 对于加载预训练模型的要单独评估
# edvr_m_wo_tsa  190.68218	windows训练时 评估

# edvr_m_wo_tsa	9.89023	0	mac训练
# edvr_m_wo_tsa	7.59464	0	mac评估 对于加载预训练模型的要单独评估
# edvr_m_wo_tsa  7.59464	mac训练时 评估

if [[ ${model} == 'ugatit_photo2cartoon' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'realsr_kernel_noise_x4_dped' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'pix2pix_cityscapes_2gpus' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'pix2pix_facades' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
elif [[ ${model} == 'firstorder_vox_mobile_256' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'pan_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'drn_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'esrgan_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'animeganv2_pretrain' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'animeganv2' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'basicvsr_reds' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'edvr_m_w_tsa' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'esrgan_x4_div2k' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
elif [[ ${model} == 'singan_finetune' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'pix2pix_cityscapes' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
elif [[ ${model} == 'firstorder_fashion' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
elif [[ ${model} == 'mprnet_deraining' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'dcgan_mnist' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'basicvsr++_reds' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'lapstyle_draft' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'singan_sr' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'cyclegan_horse2zebra' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
elif [[ ${model} == 'esrgan_psnr_x2_div2k' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
elif [[ ${model} == 'wgan_mnist' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'cond_dcgan_mnist' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'photopen' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'basicvsr++_vimeo90k_BD' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'lesrcnn_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'ugatit_selfie2anime_light' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'edvr_l_wo_tsa' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'realsr_bicubic_noise_x4_df2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'lapstyle_rev_second' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'starganv2_afhq' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'msvsr_reds' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
elif [[ ${model} == 'msvsr_vimeo90k_BD' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'iconvsr_reds' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'edvr_l_w_tsa' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'lapstyle_rev_first' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'cyclegan_cityscapes' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'singan_universal' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'singan_animation' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'starganv2_celeba_hq' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'mprnet_denoising' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
elif [[ ${model} == 'firstorder_vox_256' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
# elif [[ ${model} == 'ViT_small_patch16_224' ]]; then
#     sed -i "" "s|P0|P1|g" ${model}.yaml
#     sed -i "" "s|6.43611|0.0|g" ${model}.yaml #linux_train_单卡
#     sed -i "" "s|6.62874|0.0|g" ${model}.yaml #linux_train_多卡
#     sed -i "" "s|0.93207|0.0|g" ${model}.yaml #linux_eval
#     sed -i "" "s|16.04218|0.0|g" ${model}.yaml #linux_train_eval

#     sed -i "" "s|6.81317|0.0|g" ${model}.yaml #windows_train
#     sed -i "" "s|0.93661|0.0|g" ${model}.yaml #windows_eval
#     sed -i "" "s|190.68218|0.0|g" ${model}.yaml #windows_train_eval

#     sed -i "" "s|loss|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真
#     sed -i "" "s|train_eval|exit_code|g" ${model}.yaml # windows 训练、训练后评估都报错，暂时增加豁免为退出码为真

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
