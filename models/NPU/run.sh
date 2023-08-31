#!/bin/bash
set -x

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function printmsg()
{
    model_name=$1
    config_file=$2
    msg="${model_name} ${config_file} time cost > ${time_out}seconds"
    echo $msg >> TIMEOUT
}

function run()
{
    ps -ef | grep test_tipc | grep -v grep | cut -c 9-15 | xargs kill -9
    ps -ef | grep python | grep -v grep | cut -c 9-15 | xargs kill -9
    waitfor=7200
    command=$*
    $command &
    commandpid=$!
    ( sleep $waitfor ; kill -9 ${commandpid} >/dev/null 2>&1 && printmsg $5 $2 ) &
    watchdog=$!
    wait $commandpid >/dev/null 2>&1
    kill -9 $watchdog  >/dev/null 2>&1
}

function run_model()
{
    config_file=$1
    mode=lite_train_lite_infer
        bash test_tipc/prepare.sh $config_file $mode
        last_status=${PIPESTATUS[0]}
        if [[ ${last_status} -ne 0 ]]
        then
           exit ${last_status}
        fi
        bash test_tipc/test_train_inference_python_npu.sh $config_file $mode
        watchcat=$!
        #kill -9 -${watchcat}
}

repo=$1
ROOT_PATH=`pwd`
export FLAGS_npu_storage_format=false

if [[ ${repo} == "PaddleVideo" ]]; then
pip install opencv-python==4.5.5.62
sudo apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
cd ${ROOT_PATH}/decord/python
pwd=$PWD
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc
source ~/.bashrc
python3.7 setup.py install --user
#mv build build.bak
#mkdir build && cd build
#cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
#make
python3.7 -c 'import decord; print(decord.__version__)'
fi
cd ${ROOT_PATH}

#git clone https://github.com/PaddlePaddle/${repo}.git -b develop
if [[ ${repo} == "PaddleOCR" ]]; then
    sudo apt-get install -y libxml2-dev libxslt1-dev
    rm -rf ${repo}-release-2.6.tar.gz 
    rm -rf ${repo}
    rm -rf ${repo}-release-2.6
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-release-2.6.tar.gz
    tar -zxf ${repo}-release-2.6.tar.gz
    mv ${repo}-release-2.6 ${repo}
elif [[ ${repo} == "PaddleRec" ]]; then
    rm -rf ${repo}-master.tar.gz
    rm -rf ${repo}
    rm -rf ${repo}-master
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-master.tar.gz
    tar -zxf ${repo}-master.tar.gz
    mv ${repo}-master ${repo}
elif [[ ${repo} == "PaddleSeg" ]]; then
    rm -rf ${repo}-release-2.8.tar.gz
    rm -rf ${repo}
    rm -rf ${repo}-develop
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-release-2.8.tar.gz
    tar -zxf ${repo}-release-2.8.tar.gz
    mv ${repo}-release-2.8 ${repo}
elif [[ ${repo} == "PaddleClas" ]]; then
    rm -rf ${repo}-release-2.5.tar.gz
    rm -rf ${repo}
    rm -rf ${repo}-develop
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-release-2.5.tar.gz
    tar -zxf ${repo}-release-2.5.tar.gz
    mv ${repo}-release-2.5 ${repo}
elif [[ ${repo} == "PaddleVideo" ]]; then
    rm -rf ${repo}/test_tipc/output/*
    cd ${repo}
    git pull
    cd -
    #git clone https://github.com/PaddlePaddle/${repo}.git -b develop
elif [[ ${repo} == "PaddleDetection" ]]; then
    rm -rf ${repo}-release-2.6.tar.gz
    rm -rf ${repo}
    rm -rf ${repo}-release-2.6
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-release-2.6.tar.gz
    tar -zxf ${repo}-release-2.6.tar.gz
    mv ${repo}-release-2.6 ${repo}
else
    touch full_chain_list_all
    rm -rf ${repo}-develop.tar.gz
    rm -rf ${repo}
    rm -rf ${repo}-develop
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-develop.tar.gz
    tar -zxf ${repo}-develop.tar.gz
    mv ${repo}-develop ${repo}
fi

cd ${repo}
#if [[ ${repo} == "PaddleOCR" ]]; then
#  git checkout dygraph
#fi

if [[ ${repo} == "PaddleRec" ]]; then
#pip install --upgrade -r requirements.txt 
pip install opencv-python==4.6.0.66
pip install sklearn==0.0
pip install pandas
pip install scipy
pip install numba
#pip install pgl==2.2.4
pip install tqdm
pip install pyyaml
pip install requests
pip install nltk
pip install h5py
#pip install faiss-cpu
#pip install --use-pep517 faiss-gpu
#python setup.py install
elif [[ ${repo} == "PaddleDetection" ]]; then
pip install numpy\<1.24
pip install tqdm
pip install typeguard
pip install visualdl\>=2.2.0
pip install opencv-python\<=4.6.0
pip install PyYAML
pip install shapely
pip install scipy
pip install terminaltables
pip install Cython
pip install pycocotools
pip install setuptools
pip install lap
pip install motmetrics
pip install sklearn==0.0
pip install pyclipper
pip install Pillow==9.4.0
python setup.py install
export LD_PRELOAD=/opt/py39/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
else
pip install --upgrade -r requirements.txt
python setup.py install
echo ""
fi

if [[ ${repo} == "PaddleVideo" ]]; then
pip install av
pip install tqdm
pip install numpy==1.20
export LD_PRELOAD=/opt/compiler/gcc-8.2/lib64/libgomp.so.1
fi

if [[ ${repo} == "PaddleGAN" ]]; then
  export LD_PRELOAD=/opt/py39/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
fi

if [[ ${repo} == "PaddleNLP" ]]; then
  #python -m pip install -e ./ppdiffusers
  pip install --upgrade ppdiffusers
  cd tests
  #apt-get install -y liblzma-dev  python-lzma
  #python -m pip install --retries 10  backports.lzma
  #sed -i '27c from backports.lzma import *' /usr/local/lib/python3.9/lzma.py
  #sed -i '28c from backports.lzma import _encode_filter_properties, _decode_filter_properties' /usr/local/lib/python3.9/lzma.py
  export LD_PRELOAD=/opt/py39/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
fi
rm -rf test_tipc/output/*

sed -i 's/wget /wget -nv /g' test_tipc/prepare.sh
cp ${ROOT_PATH}/model_list.py ./
cp -r ${ROOT_PATH}/configs/ ./
cp ${ROOT_PATH}/report.py ./


if [[ ${repo} == "PaddleOCR" ]]
then
    export CUSTOM_DEVICE_BLACK_LIST=inverse
    sed -i 's#https://github.com/LDOUBLEV/AutoLog.git#https://gitee.com/Double_V/AutoLog#g' deploy/cpp_infer/external-cmake/auto-log.cmake
    sed -i 's#https://github.com/LDOUBLEV/AutoLog.git#https://gitee.com/Double_V/AutoLog#g' test_tipc/prepare.sh
    python -m pip install --retries 10 yacs
    python -m pip install --retries 10 seqeval
    export LD_PRELOAD=/opt/compiler/gcc-8.2/lib64/libgomp.so.1
    #wget https://xly-devops.bj.bcebos.com/PaddleTest/PaddleNLP/PaddleNLP-develop.tar.gz
    #tar -zxf PaddleNLP-develop.tar.gz
    #mv PaddleNLP-develop PaddleNLP
    #cd PaddleNLP
    #python -m pip install -r requirements.txt
    #python -m pip install -v -e .
    #python -m pip install -e ./ppdiffusers
    pip install --upgrade ppdiffusers
    python -m pip install Pillow==9.5.0
    #cd -
elif [[ ${repo} == "PaddleClas" ]]
then
    sed -i 's#https://github.com/LDOUBLEV/AutoLog.git#https://gitee.com/Double_V/AutoLog#g' deploy/cpp/external-cmake/auto-log.cmake
    # wget https://xly-devops.bj.bcebos.com/PaddleTest/PaddleClas/PaddleClas-develop.tar.gz
    # tar -zxf PaddleClas-develop.tar.gz
    # mv PaddleClas-develop PaddleClas
    python -m pip install -r requirements.txt
    python setup.py install
    python -m pip install -v -e .

else
echo ""
fi

touch TIMEOUT

# 确定套件的待测模型列表, 其txt保存到full_chain_list_all
if [[ ${repo} == "PaddleRec" ]]; then
#rec
#grep_models='aitm|autoint|deepfm|dlrm|dpin|dselect_k|dsin|ensfm|esmm|fgcnn|iprec|kim|mmoe|ple|tisas|wide_deep|sign|dnn'
#grep_models='aitm|autoint|deepfm|dlrm|dpin|dselect_k|dsin|ensfm|esmm|fgcnn|iprec|kim|mmoe|ple|tisas|wide_deep'
grep_models='aitm|autoint|dlrm|dpin|dselect_k|ensfm|fgcnn|mmoe|ple|tisas|wide_deep|kim'
elif [[ ${repo} == "PaddleOCR" ]]; then
#ocr
#ch_ppocr_mobile_v2_0_det
#20230721 PaddleOCR增加这三个模型ch_PP-OCRv2_rec, ch_ppocr_mobile_v2_0_rec, ch_ppocr_server_v2_0_rec，把这个模型ch_ppocr_mobile_v2_0_det_FPGM从PaddleOCR的model list删除，npu暂时不支持量化的模型
#grep_models='ch_PP-OCRv2_det|ch_PP-OCRv2_rec|ch_PP-OCRv3_det|ch_ppocr_mobile_v2_0_det|ch_ppocr_mobile_v2_0_rec|ch_ppocr_server_v2_0_det|det_mv3_db_v2_0|det_mv3_east_v2_0|det_mv3_pse_v2_0|det_r18_ct|det_r50_db_plusplus|det_r50_db_v2_0|det_r50_dcn_fce_ctw_v2_0|det_r50_vd_east_v2_0|det_r50_vd_pse_v2_0|det_r50_vd_sast_icdar15_v2_0|en_table_structure|rec_mv3_none_bilstm_ctc_v2_0|rec_mv3_none_none_ctc_v2_0|rec_vitstr'
grep_models='ch_PP-OCRv2_det|ch_PP-OCRv3_det|ch_ppocr_server_v2_0_det|det_mv3_db_v2_0|det_mv3_east_v2_0|det_mv3_pse_v2_0|det_r18_ct|det_r50_db_plusplus|det_r50_db_v2_0|det_r50_dcn_fce_ctw_v2_0|det_r50_vd_east_v2_0|det_r50_vd_pse_v2_0|det_r50_vd_sast_icdar15_v2_0|en_table_structure|rec_mv3_none_none_ctc_v2_0|rec_mtb_nrtr|rec_mv3_none_bilstm_ctc_v2_0|rec_mv3_tps_bilstm_att_v2_0|rec_mv3_tps_bilstm_ctc_v2_0|rec_r31_robustscanner|rec_r31_sar|rec_r34_vd_none_bilstm_ctc_v2_0|rec_r34_vd_none_none_ctc_v2_0|rec_r34_vd_tps_bilstm_att_v2_0|rec_r34_vd_tps_bilstm_ctc_v2_0|rec_r45_visionlan|rec_resnet_rfl|rec_svtrnet|rec_vitstr|slanet|ch_PP-OCRv2_rec|ch_ppocr_mobile_v2_0_rec|ch_ppocr_server_v2_0_rec'
elif [[ ${repo} == "PaddleGAN" ]]; then
#gan
#grep_models='Pix2pix|FOMM|edvr|basicvsr|singan|esrgan|msvsr'
grep_models='Pix2pix|edvr|esrgan|msvsr'
elif [[ ${repo} == "PaddleSeg" ]]; then
#grep_models='bisenetv1|ccnet|deeplabv3p_resnet50|encnet|enet|espnetv2|fcn_hrnetw18|fcn_hrnetw18_small|fcn_uhrnetw18_small|glore|hrnet_w48_contrast|mobileseg_mv3|ocrnet_hrnetw18|ocrnet_hrnetw48|pfpnnet|pp_liteseg_stdc1|pp_liteseg_stdc2|pphumanseg_lite|ppmatting|stdc_stdc1'
grep_models='bisenetv1|deeplabv3p_resnet50|enet|espnetv2|fcn_hrnetw18|fcn_hrnetw18_small|fcn_uhrnetw18_small|glore|hrnet_w48_contrast|mobileseg_mv3|ocrnet_hrnetw18|ocrnet_hrnetw48|pfpnnet|pp_liteseg_stdc1|pp_liteseg_stdc2|stdc_stdc1|deeplabv3p_resnet50_cityscapes|fastscnn|segformer_b0|upernet|ccnet|pphumanseg_lite|knet|ocrnet_hrformer_base|ocrnet_hrformer_small|psa|rtformer|sfnet'
grep_v_models='bisenetv2'
elif [[ ${repo} == "PaddleDetection" ]]; then
grep_models='blazeface_1000e|dark_hrnet_w32_256x192|deformable_detr_r50_1x_coco|detr_r50_1x_coco|fairmot_dla34_30e_1088x608|fairmot_hrnetv2_w18_dlafpn_30e_576x320|fcos_r50_fpn_1x_coco|gfl_r50_fpn_1x_coco|higherhrnet_hrnet_w32_512|hrnet_w32_256x192|jde_darknet53_30e_1088x608|picodet_lcnet_1_5x_416_coco|picodet_s_320_coco|picodet_s_320_coco_lcnet|ppyolo_mbv3_large_coco|ppyolo_r50vd_dcn_1x_coco|ppyolo_tiny_650e_coco|ppyoloe_crn_s_300e_coco|ppyoloe_plus_crn_s_80e_coco|ppyoloe_vit_base_csppan_cae_36e_coco|ppyolov2_r50vd_dcn_365e_coco|solov2_r50_enhance_coco|solov2_r50_fpn_1x_coco|ssdlite_mobilenet_v1_300_coco|tinypose_128x96|ttfnet_darknet53_1x_coco|yolov3_darknet53_270e_coco|yolox_s_300e_coco'
#grep_models='cascade_mask_rcnn_r50_fpn_1x_coco'
elif [[ ${repo} == "PaddleVideo" ]]; then
grep_models='BMN|SlowFast|TSM|TSN|STGCN|PP-TSM|PP-TSN|PoseC3D|AGCN|AGCN2s|TimeSformer'
elif [[ ${repo} == "PaddleNLP" ]]; then
#grep_models='bigru_crf|ernie_text_cls|ernie_text_matching|ernie_tiny|transformer'
#grep_models='bigru_crf|ernie_text_cls|bert_base_text_cls|ernie_information_extraction
grep_models='bigru_crf|ernie_information_extraction|bert_base_text_cls|bert_for_question_answering|ernie_text_cls|ernie_text_matching|ernie_tiny|ernie_information_extraction|ernie3_for_sequence_classification|seq2seq|xlnet'
elif [[ ${repo} == "PaddleClas" ]]; then
#grep_models='EfficientNetB4|EfficientNetB6|MobileViT_S|MobileViT_XS|MobileViT_XXS|PVT_V2_B0|PVT_V2_B1|PVT_V2_B2|PVT_V2_B2_Linear|PVT_V2_B3|PVT_V2_B4|PVT_V2_B5|ReXNet_1_0|ReXNet_1_3|ReXNet_1_5|ReXNet_2_0|ReXNet_3_0|SwinTransformer_base_patch4_window12_384|SwinTransformer_base_patch4_window7_224|SwinTransformer_large_patch4_window12_384|SwinTransformer_large_patch4_window7_224|SwinTransformer_small_patch4_window7_224|SwinTransformer_tiny_patch4_window7_224|alt_gvt_base|alt_gvt_large|alt_gvt_small|pcpvt_base|pcpvt_large|pcpvt_small|ViT_base_patch16_224|ViT_base_patch32_384|ViT_large_patch16_224|ViT_small_patch16_224'
#grep_models='PVT_V2_B1|PVT_V2_B2|PVT_V2_B2_Linear|PVT_V2_B3|PVT_V2_B4|PVT_V2_B5|ReXNet_1_0|ReXNet_1_3|ReXNet_1_5|ReXNet_2_0|ReXNet_3_0|SwinTransformer_base_patch4_window12_384|SwinTransformer_base_patch4_window7_224|SwinTransformer_large_patch4_window12_384|SwinTransformer_large_patch4_window7_224|alt_gvt_base|alt_gvt_large|alt_gvt_small|pcpvt_base|pcpvt_large|pcpvt_small|ViT_small_patch16_224'
grep_models='AlexNet|CSWinTransformer_base_224|ConvNeXt_tiny|DLA102|DLA169|DLA34|DLA46_c|DLA60|DPN107|DPN131|DPN92|DPN98|DeiT_base_patch16_224|DenseNet121|ESNet_x0_25|EfficientNetB0|GhostNet_x0_5|GoogLeNet|HRNet_W18_C|HarDNet39_ds|InceptionV3|LeViT_128|MobileNetV1|MobileNetV2|MobileNetV3_small_x0_5|MobileViT_S|MobileViT_XS|MobileViT_XXS|PPHGNet_tiny|PPLCNetV2_base|PPLCNet_x0_5|PPLCNet_x1_0|PVT_V2_B0|PVT_V2_B2_Linear|Res2Net101_vd_26w_4s|ResNeSt50_fast_1s1x64d|ResNeXt101_32x4d|ResNeXt101_vd_32x4d|ResNeXt152_64x4d|ResNeXt152_vd_32x4d|ResNeXt50_32x4d|ResNeXt50_vd_32x4d|ResNet101|ResNet101_vd|SENet154_vd|SE_ResNeXt50_32x4d|SE_ResNeXt50_vd_32x4d|SE_ResNet18_vd|ShuffleNetV2_swish|ShuffleNetV2_x0_25|SqueezeNet1_0|SwinTransformer_base_patch4_window12_384|VGG11|ViT_base_patch16_224|Xception41|alt_gvt_base|pcpvt_base|RedNet26|ReXNet_1_3'
else
grep_models=''
fi

touch full_chain_list_all_tmp
touch full_chain_list_all
mode=lite_train_lite_infer
time_out=3600
if [[ ${repo} == "PaddleDetection" ]]; then
    time_out=7200
fi

if [[ ${repo} == "PaddleClas" ]]; then
    time_out=7200
fi

file_txt=*train_infer_python.txt*
python model_list.py $repo ${PWD}/test_tipc/configs/ $file_txt full_chain_list_all_tmp

#0713新增 删除运行后数据
rm -rf core.*
rm -rf kernel_meta*
rm -rf test_tipc/output/*
#==========================
if [ ! ${grep_models} ]; then
    grep_models=undefined
fi
if [ ! ${grep_v_models} ]; then
    grep_v_models=undefined
fi
if [[ ${grep_models} =~ "undefined" ]]; then
    if [[ ${grep_v_models} =~ "undefined" ]]; then
        cat full_chain_list_all_tmp | sort | uniq > full_chain_list_all
    else
        cat full_chain_list_all_tmp | sort | uniq |grep -v -E ${grep_v_models} > full_chain_list_all  #除了剔除的都跑
    fi
else
    if [[ ${grep_v_models} =~ "undefined" ]]; then
        cat full_chain_list_all_tmp | sort | uniq | grep -E ${grep_models} > full_chain_list_all
    else
        cat full_chain_list_all_tmp | sort | uniq |grep -v -E ${grep_v_models} |grep -E ${grep_models} > full_chain_list_all
    fi
fi
cat full_chain_list_all

cat full_chain_list_all | while read config_file
do
  start=`date +%s`
  echo "==START=="$config_file
  
  dataline=$(awk 'NR==1, NR==32{print}'  $config_file)
  IFS=$'\n'
  lines=(${dataline})
  model_name=$(func_parser_value "${lines[1]}")

  sleep 10
  run run_model $config_file $mode $time_out $model_name
  sleep 10
  #bash test_tipc/prepare.sh $config_file $mode
  #bash test_tipc/test_train_inference_python_npu.sh $config_file $mode

  echo "==END=="$config_file
  end=`date +%s`
  time=`echo $start $end | awk '{print $2-$1-2}'`
  echo "${config_file} spend time seconds ${time}"
done

# get cmd RESULT
log_file="RESULT"
>${log_file}
current_path=`pwd`
echo ${current_path}
echo ${repo}
if [[ ${current_path} == *${repo}* ]]
then
for f in `find . -name '*.log'`; do
   echo $f
   cat $f | grep "with command" >> $log_file
done
fi

#echo "THE RESULT ${repo}" >>${ROOT_PATH}/FINAL_RESULT_${repo}

#echo "" >>${ROOT_PATH}/FINAL_RESULT_${repo}
python report.py ${repo} chain_base npu_test@baidu.com suijiaxin@baidu.com,songkai05@baidu.com,duanyanhui@baidu.com,liqi27@baidu.com proxy-in.baidu.com >${ROOT_PATH}/FINAL_RESULT_${repo}
#python report.py ${repo} chain_base npu_test@baidu.com suijiaxin@baidu.com proxy-in.baidu.com >${ROOT_PATH}/FINAL_RESULT_${repo}
