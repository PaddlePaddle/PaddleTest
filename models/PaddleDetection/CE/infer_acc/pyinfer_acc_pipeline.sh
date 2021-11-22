#!/bin/bash

echo -e '*****************paddle_version*****'
python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************detection_version****'
git rev-parse HEAD

python -m pip install pip==20.2.4 --ignore-installed;
pip install Cython --ignore-installed;
pip install -r requirements.txt --ignore-installed;
pip install cython_bbox --ignore-installed;
apt-get update && apt-get install -y ffmpeg
if [ -d 'dataset/coco' ];then
rm -rf dataset/coco
fi
ln -s ${data_path}/coco dataset/coco
if [ -d 'dataset/voc' ];then
rm -rf dataset/voc
fi
ln -s ${data_path}/../PaddleSeg/pascalvoc dataset/voc
if [ -d "dataset/mot" ];then rm -rf dataset/mot
fi
ln -s ${data_path}/mot dataset/mot
if [ -d "dataset/mpii" ];then rm -rf dataset/mpii
fi
ln -s ${data_path}/mpii_tar dataset/mpii
if [ -d "dataset/DOTA_1024_s2anet" ];then rm -rf dataset/DOTA_1024_s2anet
fi
ln -s ${data_path}/DOTA_1024_s2anet dataset/DOTA_1024_s2anet
if [ -d "dataset/VisDrone2019_coco" ];then rm -rf dataset/VisDrone2019_coco
fi
ln -s ${data_path}/VisDrone2019_coco dataset/VisDrone2019_coco

if [ -d "log" ];then
rm -rf log
fi
mkdir log
if [ -d "log_err" ];then
rm -rf log_err
fi
mkdir log_err

sed -i "140d" deploy/python/visualize.py
sed -i "140d" deploy/python/visualize.py
sed -i "140d" deploy/python/visualize.py
sed -i "/            xmin, ymin, xmax, ymax = bbox/a\            print('class_id:{:d} confidence:{:.4f} box:{:.2f},{:.2f},{:.2f},{:.2f}'.format(int(clsid), score, xmin, ymin, xmax, ymax))" deploy/python/visualize.py
sed -i '/                print("save result to: " + out_path)/a\        results.append((frame_id + 1, online_tlwhs, online_scores, online_ids))' deploy/python/mot_jde_infer.py
sed -i '/        results.append((frame_id + 1, online_tlwhs, online_scores, online_ids))/a\        if FLAGS.save_mot_txt_per_img:' deploy/python/mot_jde_infer.py
sed -i '/        if FLAGS.save_mot_txt_per_img:/a\            img_file_name = img_file.replace("/","_").replace("jpg","txt")' deploy/python/mot_jde_infer.py
sed -i '/^            img_file_name = img_file.replace/a\            save_dir = FLAGS.output_dir' deploy/python/mot_jde_infer.py
sed -i '/            save_dir = FLAGS.output_dir/a\            if not os.path.exists(save_dir):os.makedirs(save_dir)' deploy/python/mot_jde_infer.py
sed -i '/            if not os.path.exists(save_dir):os.makedirs(save_dir)/a\            result_filename = os.path.join(save_dir,"\{\}.txt".format(img_file_name))' deploy/python/mot_jde_infer.py
sed -i '/            result_filename = os.path.join(save_dir,"{}.txt".format(img_file_name))/a\            write_mot_results(result_filename, [results[-1]])' deploy/python/mot_jde_infer.py

#prepare det model of unite predict

print_result(){
    if [ $? -ne 0 ];then
        echo -e "${model},${mode},FAIL"
        err_sign=true
        cd log_err
        if [ ! -d ${model} ];then
            mkdir ${model}
        fi
        cd ../${model_path}
        mv log/${model}/${model}_${mode}.log log_err/${model}/
    else
        echo -e "${model},${mode},SUCCESS"
    fi
}
EXPORT_DET(){
    if [ ! -f /root/.cache/paddle/weights/ppyolov2_r50vd_dcn_365e_coco.pdparams ];then
        wget -P /root/.cache/paddle/weights/ https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams
        if [ ! -s /root/.cache/paddle/weights/ppyolov2_r50vd_dcn_365e_coco.pdparams ];then
            echo "ppyolov2 url is bad,so can't run EXPORT!"
        else
            mode=ppyolov2_export
            python tools/export_model.py \
                   -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml \
                   --output_dir=inference_model \
                   -o weights=https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams >log/${model}/${mode}.log 2>&1
            print_result
        fi
    else
        mode=ppyolov2_export
        python tools/export_model.py \
               -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml \
               --output_dir=inference_model \
               -o weights=https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams >log/${model}/${mode}.log 2>&1
        print_result
    fi
}
EXPORT(){
    if [ ! -f /root/.cache/paddle/weights/${model}.pdparams ];then
        wget -P /root/.cache/paddle/weights/ https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams
        if [ ! -s /root/.cache/paddle/weights/${model}.pdparams ];then
            echo "${model} url is bad,so can't run EXPORT!"
        else
            mode=export
            python tools/export_model.py \
                   -c ${config} \
                   --output_dir=inference_model \
                   -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${mode}.log 2>&1
            print_result
        fi
    else
        mode=export
        python tools/export_model.py \
               -c ${config} \
               --output_dir=inference_model \
               -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}

PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run PYTHON_INFER case!"
    else
        mode=python_infer
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python deploy/python/infer.py \
               --model_dir=inference_model/${model} \
               --image_file=demo/000000014439.jpg \
               --run_mode=fluid \
               --device=GPU \
               --threshold=0.5 \
               --output_dir=python_infer_output/${model}/ >log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
MOT_PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run MOT_PYTHON_INFER case!"
    else
        mode=mot_python_infer
        export PYTHONPATH=`pwd`
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python deploy/python/mot_jde_infer.py \
               --model_dir=./inference_model/${model} \
               --image_file=demo/000000014439.jpg \
               --device=GPU \
               --save_mot_txt_per_img \
               --save_images \
               --output_dir=output/${model} >log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
POSE_UNITE_PYTHON_INFER(){
    if [ ! -d ./inference_model/${model} ];then
        echo "${model} doesn't run export case,so can't run POSE_PYTHON_INFER case!"
    else
        mode=pose_python_infer
        export CUDA_VISIBLE_DEVICES=$cudaid1
        python deploy/python/det_keypoint_unite_infer.py \
               --det_model_dir=inference_model/ppyolov2_r50vd_dcn_365e_coco/ \
               --keypoint_model_dir=inference_model/${model} \
               --image_file=demo/000000014439.jpg \
               --device=GPU \
               --run_mode=fluid \
               --save_res=True \
               --output_dir=python_infer_output/${model} >log/${model}/${model}_${mode}.log 2>&1
        print_result
    fi
}
weight_dir=
err_sign=false
for config in `cat model_acc_list`
do
tmp=${config##*/}
model=${tmp%.*}
cd log && mkdir ${model} && cd ..
if [[ -n `echo "${config}" | grep "keypoint"` ]];then
weight_dir=keypoint/
rm -f det_keypoint_unite_video_results.json
EXPORT_DET
EXPORT
POSE_UNITE_PYTHON_INFER
python  python_infer_accuracy.py --model_type="keypoint" --sample_log='det_keypoint_unite_image_results2.json' --run_log='det_keypoint_unite_image_results.json' --model_name=${model}
elif [[ -n `echo "${config}" | grep "mot"` ]];then
weight_dir=mot/
EXPORT
MOT_PYTHON_INFER
python  python_infer_accuracy.py --model_type="mot" --sample_log='pyinfer_sample.txt' --run_log=output/${model}/demo_000000014439.txt.txt --model_name=${model}
else
EXPORT
PYTHON_INFER
python  python_infer_accuracy.py --model_type="det" --sample_log='pyinfer_sample.txt' --run_log=log/${model}/${model}_${mode}.log --model_name=${model}
fi
done
if [ "${err_sign}" == true ];then
    exit 1
fi

