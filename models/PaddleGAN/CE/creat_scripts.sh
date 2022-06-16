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

# edvr_m_wo_tsa	loss_pixel	指标
# edvr_m_wo_tsa	8462.812	linux单卡训练
# edvr_m_wo_tsa	9151.643	linux多卡训练

if [[ ${model} == 'ugatit_photo2cartoon' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|discriminator_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|2.724|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|2.214|g" ${model}.yaml #linux_train_单卡

elif [[ ${model} == 'realsr_kernel_noise_x4_dped' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pix|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|1.143|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|1.986|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'pix2pix_cityscapes_2gpus' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|D_fake_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.695|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.518|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'pix2pix_facades' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|D_fake_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.759|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.639|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'firstorder_vox_mobile_256' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|perceptual|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|312.392|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|226.539|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'pan_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.136|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.122|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'drn_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_dual|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|106.452|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|109.42|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'esrgan_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|28.251|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|15.114|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'animeganv2_pretrain' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|init_c_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|672.322|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|611.257|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'animeganv2' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|d_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|621.378|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|625.417|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'basicvsr_reds' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.052|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.044|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'edvr_m_w_tsa' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|9246.57|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|8703.523|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'esrgan_x4_div2k' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pix|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.864|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.872|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'singan_finetune' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|scale1/D_gradient_penalty|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.156|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.156|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'pix2pix_cityscapes' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
    sed -i "" "s|loss_pixel|D_fake_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.824|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.519|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'firstorder_fashion' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
    sed -i "" "s|loss_pixel|perceptual|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|196.501|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|176.988|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'mprnet_deraining' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|44290.164|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|48742.102|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'dcgan_mnist' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|training_exit_code|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'basicvsr++_reds' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.062|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.05|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'lapstyle_draft' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_c|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|5.823|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|5.823|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'singan_sr' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|scale9/D_gradient_penalty|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.219|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.219|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'cyclegan_horse2zebra' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
    sed -i "" "s|loss_pixel|G_idt_A_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|2.856|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|2.874|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'esrgan_psnr_x2_div2k' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|26.395|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|14.98|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'wgan_mnist' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|training_exit_code|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'cond_dcgan_mnist' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|training_exit_code|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'photopen' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|g_featloss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|12.787|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|12.787|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'basicvsr++_vimeo90k_BD' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.022|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.036|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'lesrcnn_psnr_x4_div2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.455|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.388|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'ugatit_selfie2anime_light' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|discriminator_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|4.439|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|4.995|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'edvr_l_wo_tsa' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|9268.696|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|9860.704|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'realsr_bicubic_noise_x4_df2k' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pix|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.269|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.27|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'lapstyle_rev_second' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|D_fake_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.702|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.702|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'starganv2_afhq' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|G/latent_adv|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|108.868|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|108.868|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'msvsr_reds' ]]; then
    sed -i "" "s|P0|P0|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pix|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.062|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.054|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'msvsr_vimeo90k_BD' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pix|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.026|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.031|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'iconvsr_reds' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.064|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.058|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'edvr_l_w_tsa' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss_pixel|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|10505.313|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|6680.915|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'lapstyle_rev_first' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|D_fake_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.708|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.708|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'cyclegan_cityscapes' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|G_idt_A_loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|1.568|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|1.186|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'singan_universal' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|scale6/D_gradient_penalty|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.206|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.206|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'singan_animation' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|scale6/D_gradient_penalty|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0.269|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0.269|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'starganv2_celeba_hq' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|G/latent_adv|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|27.779|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|7.542|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'mprnet_denoising' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|loss|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|16058.466|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|19116.625|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'firstorder_vox_256' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|perceptual|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|258.835|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|194.289|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'stylegan_v2_256_ffhq' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|l_d|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|2.398|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|2.726|g" ${model}.yaml #linux_train_多卡

elif [[ ${model} == 'makeup' ]]; then
    sed -i "" "s|P0|P1|g" ${model}.yaml
    sed -i "" "s|loss_pixel|training_exit_code|g" ${model}.yaml #指标
    sed -i "" "s|8462.812|0|g" ${model}.yaml #linux_train_单卡
    sed -i "" "s|9151.643|0|g" ${model}.yaml #linux_train_多卡

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
