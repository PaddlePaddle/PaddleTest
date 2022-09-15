#!/usr/bin/env bash

python tools/export_model.py -c configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml --output_dir=./inference_model \
 -o weights=pretrain_model/cascade_rcnn_r50_fpn_1x_coco

python tools/infer.py -c configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml --infer_img=demo/000000087038.jpg \
  -o weights=pretrain_model/cascade_rcnn_r50_fpn_1x_coco

python tools/export_model.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml --output_dir=./inference_model \
 -o weights=pretrain_model/ppyolo_r50vd_dcn_1x_coco

python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml --infer_img=demo/000000087038.jpg \
  -o weights=pretrain_model/ppyolo_r50vd_dcn_1x_coco

python tools/export_model.py -c configs/centernet/centernet_dla34_140e_coco.yml --output_dir=./inference_model \
 -o weights=pretrain_model/centernet_dla34_140e_coco

python tools/train.py -c configs/gn/faster_rcnn_r50_fpn_gn_2x_coco.yml

python tools/train.py -c configs/detr/detr_r50_1x_coco.yml

python tools/train.py -c configs/deformable_detr/deformable_detr_r50_1x_coco.yml

python tools/train.py -c configs/sparse_rcnn/sparse_rcnn_r50_fpn_3x_pro100_coco.yml -o TrainReader.batch_size=1

python tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o TrainReader.batch_size=1

python tools/train.py -c configs/res2net/mask_rcnn_res2net50_vb_26w_4s_fpn_2x_coco.yml -o TrainReader.batch_size=1

python tools/train.py -c configs/ppyolo/ppyolo_mbv3_small_coco.yml -o TrainReader.batch_size=1

python tools/train.py -c configs/dota/s2anet_1x_spine.yml -o TrainReader.batch_size=1

visualdl --logdir cascade_rcnn_r50_fpn_1x_coco --host 0.0.0.0 --port 8881

visualdl --logdir ppyolo_r50vd_dcn_1x_coco --host 0.0.0.0 --port 8881

visualdl --logdir centernet_dla34_140e_coco --host 0.0.0.0 --port 8882

https://paddledet.bj.bcebos.com/models/centernet_dla34_140e_coco.pdparams


head: BBoxHead

paddle2onnx --model_dir=ssdlite_mobilenet_v1_300_coco_upload \
        --model_filename=model.pdmodel \
        --params_filename=model.pdiparams \
        --save_file=ssdlite_mobilenet_v1_300_coco_upload/inference.onnx \
        --opset_version=12 --enable_onnx_checker=True
