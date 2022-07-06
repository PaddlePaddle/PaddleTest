#!/bin/bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
# code_path=$cur_path/../../PaddleDetection



# 准备数据
if [ -d "$cur_path/../../PaddleDetection/dataset/coco" ];then
rm -rf $cur_path/../../PaddleDetection/dataset/coco
fi
ln -s ${data_path}/coco $cur_path/../../PaddleDetection/dataset/coco
if [ -d "$cur_path/../../PaddleDetection/dataset/voc" ];then
rm -rf $cur_path/../../PaddleDetection/dataset/voc
fi
ln -s ${data_path}/pascalvoc $cur_path/../../PaddleDetection/dataset/voc
if [ -d "$cur_path/../../PaddleDetection/dataset/mot" ];then
rm -rf $cur_path/../../PaddleDetection/dataset/mot
fi
ln -s ${data_path}/mot $cur_path/../../PaddleDetection/dataset/mot
if [ -d "$cur_path/../../PaddleDetection/dataset/wider_face" ];then
rm -rf $cur_path/../../PaddleDetection/dataset/wider_face
fi
ln -s ${data_path}/wider_face $cur_path/../../PaddleDetection/dataset/wider_face
if [ -d "$cur_path/../../PaddleDetection/dataset/AIchallenge" ];then
rm -rf $cur_path/../../PaddleDetection/dataset/AIchallenge
fi
ln -s ${data_path}/AIchallenge $cur_path/../../PaddleDetection/dataset/AIchallenge
if [ -d "$cur_path/../../PaddleDetection/dataset/aic_coco_train_cocoformat.json" ];then
rm -rf $cur_path/../../PaddleDetection/dataset/aic_coco_train_cocoformat.json
fi
ln -s ${data_path}/aic_coco_train_cocoformat.json $cur_path/../../PaddleDetection/dataset/aic_coco_train_cocoformat.json

#train config process
max_iters=121 #2500 #1000
cd $cur_path/../../PaddleDetection
pip install -r requirements.txt
sed -i "/for step_id, data in enumerate(self.loader):/i\            max_step_id =${max_iters}" $cur_path/../../PaddleDetection/ppdet/engine/trainer.py
sed -i "/for step_id, data in enumerate(self.loader):/a\                if step_id == max_step_id: break" $cur_path/../../PaddleDetection/ppdet/engine/trainer.py
#cascade
sed -i "5c 5c  - RandomResize: {target_size: [[800, 1333]], interp: 2, keep_ratio: True}" $cur_path/../../PaddleDetection/configs/cascade_rcnn/_base_/cascade_fpn_reader.yml
sed -i "s/5c//" $cur_path/../../PaddleDetection/configs/cascade_rcnn/_base_/cascade_fpn_reader.yml
sed -i "6c 6c  - RandomFlip: {prob: 0.}" $cur_path/../../PaddleDetection/configs/cascade_rcnn/_base_/cascade_fpn_reader.yml
sed -i "s/6c//" $cur_path/../../PaddleDetection/configs/cascade_rcnn/_base_/cascade_fpn_reader.yml
sed -i "/CascadeRCNN:/i\find_unused_parameters: True" $cur_path/../../PaddleDetection/configs/cascade_rcnn/_base_/cascade_rcnn_r50_fpn.yml
#fcos
sed -i "/FCOS:/i\find_unused_parameters: True" $cur_path/../../PaddleDetection/configs/fcos/_base_/fcos_r50_fpn.yml
sed -i "5c 5c  - RandomFlip: {prob: 0.}" $cur_path/../../PaddleDetection/configs/fcos/_base_/fcos_reader.yml
sed -i "s/5c//" $cur_path/../../PaddleDetection/configs/fcos/_base_/fcos_reader.yml
#mask
sed -i "5c 5c  - RandomResize: {target_size: [[800, 1333]], interp: 2, keep_ratio: True}" $cur_path/../../PaddleDetection/configs/mask_rcnn/_base_/mask_fpn_reader.yml
sed -i "s/5c//" $cur_path/../../PaddleDetection/configs/mask_rcnn/_base_/mask_fpn_reader.yml
sed -i "6c 6c  - RandomFlip: {prob: 0.}" $cur_path/../../PaddleDetection/configs/mask_rcnn/_base_/mask_fpn_reader.yml
sed -i "s/6c//" $cur_path/../../PaddleDetection/configs/mask_rcnn/_base_/mask_fpn_reader.yml
sed -i "/MaskRCNN:/i\find_unused_parameters: True" $cur_path/../../PaddleDetection/configs/mask_rcnn/_base_/mask_rcnn_r50_fpn.yml
#solov2
sed -i "/SOLOv2:/i\find_unused_parameters: True" $cur_path/../../PaddleDetection/configs/solov2/_base_/solov2_r50_fpn.yml
sed -i "s/^  - RandomFlip:/#&/" $cur_path/../../PaddleDetection/configs/solov2/_base_/solov2_reader.yml
#yolov3
sed -i "13c 13c    - BatchRandomResize: {target_size: [320], random_size: True, random_interp: False, keep_ratio: False}" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
sed -i "s/13c//" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
sed -i "s/^    - Mixup:/#&/" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
sed -i "s/^    - RandomDistort:/#&/" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
sed -i "s/^    - RandomExpand:/#&/" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
sed -i "s/^    - RandomCrop:/#&/" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
sed -i "s/^    - RandomFlip:/#&/" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
sed -i "s/^  mixup_epoch/#&/" $cur_path/../../PaddleDetection/configs/yolov3/_base_/yolov3_reader.yml
#ppyolov2
sed -i "13c 13c    - BatchRandomResize: {target_size: [320], random_size: True, random_interp: False, keep_ratio: False}" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
sed -i "s/13c//" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
sed -i "s/^    - Mixup:/#&/" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
sed -i "s/^    - RandomDistort:/#&/" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
sed -i "s/^    - RandomExpand:/#&/" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
sed -i "s/^    - RandomCrop:/#&/" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
sed -i "s/^    - RandomFlip:/#&/" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
sed -i "s/^  mixup_epoch/#&/" $cur_path/../../PaddleDetection/configs/ppyolo/_base_/ppyolov2_reader.yml
#ssd
sed -i "s/^    - RandomDistort:/#&/" $cur_path/../../PaddleDetection/configs/ssd/_base_/ssd_mobilenet_reader.yml
sed -i "s/^    - RandomExpand:/#&/" $cur_path/../../PaddleDetection/configs/ssd/_base_/ssd_mobilenet_reader.yml
sed -i "s/^    - RandomCrop:/#&/" $cur_path/../../PaddleDetection/configs/ssd/_base_/ssd_mobilenet_reader.yml
sed -i "s/^    - RandomFlip:/#&/" $cur_path/../../PaddleDetection/configs/ssd/_base_/ssd_mobilenet_reader.yml
#hrnet
sed -i "97c 97c        prob_half_body: 0." $cur_path/../../PaddleDetection/configs/keypoint/hrnet/hrnet_w32_256x192.yml
sed -i "s/97c//" $cur_path/../../PaddleDetection/configs/keypoint/hrnet/hrnet_w32_256x192.yml
sed -i "/if np.random.randn() < 0.5 and len(upper_joints) > 2:/a\        if True and len(upper_joints) > 2:" $cur_path/../../PaddleDetection/ppdet/data/transform/keypoint_operators.py
sed -i "s/^        if np.random.randn() < 0.5/#&/" $cur_path/../../PaddleDetection/ppdet/data/transform/keypoint_operators.py
#ttfnet
sed -i "s/^  - RandomFlip:/#&/" $cur_path/../../PaddleDetection/configs/ttfnet/_base_/ttfnet_reader.yml
