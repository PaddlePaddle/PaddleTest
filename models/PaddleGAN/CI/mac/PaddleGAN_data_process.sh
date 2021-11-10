sed -i '' 's/epochs/total_iters/g' configs/cyclegan_cityscapes.yaml #将epcoh换为iter
sed -i '' 's/decay_total_iters/decay_epochs/g' configs/cyclegan_cityscapes.yaml #恢复学习率衰减字段
sed -i '' 's/RandomCrop/Resize/g' configs/cyclegan_cityscapes.yaml #将 RandomCrop 字段替换为 Resize
sed -ie '/- name: RandomHorizontalFlip/d' configs/cyclegan_cityscapes.yaml #删除 RandomHorizontalFlip 字段行
sed -ie '/prob: 0.5/d' configs/cyclegan_cityscapes.yaml #删除 prob 字段行

sed -ie '/- name: PairedRandomHorizontalFlip/{N;d;}' configs/esrgan_x4_div2k.yaml  #删除随机变量
sed -ie '/- name: PairedRandomVerticalFlip/{N;d;}' configs/esrgan_x4_div2k.yaml
sed -ie '/- name: PairedRandomTransposeHW/{N;d;}' configs/esrgan_x4_div2k.yaml

sed -i '' 's/use_flip: True/use_flip: False/g' configs/edvr_m_wo_tsa.yaml #将 use_flip 字段替换为 Fasle
sed -i '' 's/use_rot: True/use_rot: False/g' configs/edvr_m_wo_tsa.yaml #将 use_rot 字段替换为 Fasle

sed -i '' 's/epochs/total_iters/g' configs/firstorder_fashion.yaml #将epcoh换为iter
sed -ie '/- name: PairedRandomHorizontalFlip/d' configs/firstorder_fashion.yaml #删除 - name: PairedRandomHorizontalFlip
sed -ie '/prob: 0.5/{N;d;}' configs/firstorder_fashion.yaml  #删除随机变量 相关参数


# data
rm -rf data
ln -s /Users/paddle/PaddleTest/ce_data/PaddleGAN/small_dataset_for_CE data
