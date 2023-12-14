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
        bash test_tipc/test_train_inference_python_xpu.sh $config_file $mode
        watchcat=$!
        #kill -9 -${watchcat}
}

repo=$1
ROOT_PATH=`pwd`

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
    rm -rf ${repo}-release-2.7.tar.gz 
    rm -rf ${repo}
    rm -rf ${repo}-release-2.7
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-release-2.7.tar.gz
    tar -zxf ${repo}-release-2.7.tar.gz
    mv ${repo}-release-2.7 ${repo}
    #git clone https://github.com/plusNew001/PaddleOCR.git
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
    rm -rf ${repo}-develop.tar.gz
    rm -rf ${repo}
    rm -rf ${repo}-develop
    wget -nv https://xly-devops.bj.bcebos.com/PaddleTest/${repo}/${repo}-develop.tar.gz
    tar -zxf ${repo}-develop.tar.gz
    mv ${repo}-develop ${repo}
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
pip install numpy \< 1.24
pip install tqdm
pip install typeguard
pip install visualdl\>=2.2.0
pip install opencv-python \<= 4.6.0
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
python setup.py install
#export LD_PRELOAD=/opt/py39/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
else
pip install --upgrade -r requirements.txt
python setup.py install
echo ""
fi

if [[ ${repo} == "PaddleVideo" ]]; then
pip install av
pip install tqdm
pip install numpy
pip install pandas
pip install PyYAML>=5.1
pip install opencv-python
pip install decord==0.4.2
pip install scipy==1.6.3
pip install scikit-image
#export LD_PRELOAD=/opt/compiler/gcc-8.2/lib64/libgomp.so.1
fi

#if [[ ${repo} == "PaddleGAN" ]]; then
#  export LD_PRELOAD=/opt/py39/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
#fi

if [[ ${repo} == "PaddleNLP" ]]; then
  python -m pip install -e ./ppdiffusers
  cd tests
  #apt-get install -y liblzma-dev  python-lzma
  #python -m pip install --retries 10  backports.lzma
  #sed -i '27c from backports.lzma import *' /usr/local/lib/python3.9/lzma.py
  #sed -i '28c from backports.lzma import _encode_filter_properties, _decode_filter_properties' /usr/local/lib/python3.9/lzma.py
  #export LD_PRELOAD=/opt/py39/lib/python3.9/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
fi
rm --rf test_tipc/output/*

sed -i 's/wget /wget -nv /g' test_tipc/prepare.sh
cp ${ROOT_PATH}/model_list.py ./
cp -r ${ROOT_PATH}/configs/ ./
cp ${ROOT_PATH}/report.py ./


if [[ ${repo} == "PaddleOCR" ]]
then
    sed -i 's#https://github.com/LDOUBLEV/AutoLog.git#https://gitee.com/Double_V/AutoLog#g' deploy/cpp_infer/external-cmake/auto-log.cmake
    sed -i 's#https://github.com/LDOUBLEV/AutoLog.git#https://gitee.com/Double_V/AutoLog#g' test_tipc/prepare.sh
    python -m pip install --retries 10 yacs
    python -m pip install --retries 10 seqeval
    python -m pip install --retries 10 paddleslim
    # export LD_PRELOAD=/opt/compiler/gcc-8.2/lib64/libgomp.so.1
    wget https://xly-devops.bj.bcebos.com/PaddleTest/PaddleNLP/PaddleNLP-develop.tar.gz
    tar -zxf PaddleNLP-develop.tar.gz
    mv PaddleNLP-develop PaddleNLP
    cd PaddleNLP
    python -m pip install -r requirements.txt
    python -m pip install -v -e .
    python -m pip install -e ./ppdiffusers
    cd -
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
grep_models='deepfm|wide_deep|dlrm|aitm|dnn|mmoe|kim|ensfm'
elif [[ ${repo} == "PaddleOCR" ]]; then

grep_models='ch_PP-OCRv3|det_mv3_db_v2_0|det_mv3_east_v2_0|det_r50_vd_pse_v2_0|vi_layoutxlm_ser|en_server_pgnetA|ch_ppocr_mobile_v2_0_det|rec_r34_vd_none_none_ctc_v2_0|sr_telescope'
elif [[ ${repo} == "PaddleGAN" ]]; then
#gan
#grep_models='Pix2pix|FOMM|edvr|basicvsr|singan|esrgan|msvsr'
grep_models='Pix2pix|edvr|esrgan'
elif [[ ${repo} == "PaddleSeg" ]]; then
#grep_models='bisenetv1|ccnet|deeplabv3p_resnet50|encnet|enet|espnetv2|fcn_hrnetw18|fcn_hrnetw18_small|fcn_uhrnetw18_small|glore|hrnet_w48_contrast|mobileseg_mv3|ocrnet_hrnetw18|ocrnet_hrnetw48|pfpnnet|pp_liteseg_stdc1|pp_liteseg_stdc2|pphumanseg_lite|ppmatting|stdc_stdc1'
grep_models='ppmatting|hrnet_w48_contrast|pp_liteseg_stdc1|deeplabv3p_resnet50_cityscapes|pphumanseg_server|ocrnet_hrnetw18|bisenetv2|ddrnet|encnet|enet|sfnet'
grep_v_models='bisenetv2'
elif [[ ${repo} == "PaddleDetection" ]]; then
grep_models='ssd_r34_70e_coco|ssd_vgg16_300_240e_voc|yolov3_darknet53_270e_voc|ssd_r34_70e_coco|fairmot_hrnetv2_w18_dlafpn_30e_576x320|ppyoloe_crn_s_300e_coco|yolov3_mobilenet_v1_270e_coco|mask_rcnn_r50_1x_coco|ssdlite_mobilenet_v1_300_coco|faster_rcnn_r50_1x_coco|ppyolov2_r50vd_dcn_365e_coco|picodet_s_320_coco_lcnet|dark_hrnet_w32_256x192'
#grep_models='cascade_mask_rcnn_r50_fpn_1x_coco'
elif [[ ${repo} == "PaddleVideo" ]]; then
grep_models='PP-TSM|AGCN|BMN|PoseC3D|TSM'
elif [[ ${repo} == "PaddleNLP" ]]; then
#grep_models='bigru_crf|ernie_text_cls|ernie_text_matching|ernie_tiny|transformer'
#grep_models='bigru_crf|ernie_text_cls|bert_base_text_cls|ernie_information_extraction
grep_models='ernie_information_extraction|bert_base_text_cls|transformer|gpt2|seq2seq|xlnet|stablediffusion'
elif [[ ${repo} == "PaddleClas" ]]; then
#grep_models='MobileNetV3_small_x1_0_ampo2_ultra|MobileNetV3_small_x1_0_fp32_ultra|PPLCNet_x1_0_ampo2_ultra|PPLCNet_x1_0_fp32_ultra|PeleeNet'
grep_models='RsNet50|MobileNetV3_small_x1_0|InceptionV4|MobileNetV2_x1_5|PPLCNet_x1_0|PPHGNet_tiny|VGG16|VGG19|ResNet101|PeleeNet|DPN68|DLA102|ShuffleNetV2_swish|EfficientNetB0|DenseNet121|HarDNet39_ds|DeiT_tiny_patch16_224|InceptionV3|HRNet_W18_C|GhostNet_x0_5|ESNet_x0_25|MixNet_S|LeViT_128|TNT_small|ViT_base_patch16_224|PVT_V2_B0|CSWinTransformer_tiny_224|GoogLeNet'
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
  if [[ ${repo} == "PaddleClas" ]]; then
    sed -i '16s/$/ -o Global.use_dali=False/'  $config_file
    sed -i '24s/$/ -o Global.use_dali=False/'  $config_file
  fi
  start=`date +%s`
  echo "==START=="$config_file
  
  dataline=$(awk 'NR==1, NR==32{print}'  $config_file)
  IFS=$'\n'
  lines=(${dataline})
  model_name=$(func_parser_value "${lines[1]}")

  sleep 10
  run run_model $config_file $mode $time_out $model_name
  sleep 10


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
# python report.py ${repo} chain_base xpu_test@baidu.com suijiaxin@baidu.com,songkai05@baidu.com,liqi27@baidu.com proxy-in.baidu.com >${ROOT_PATH}/FINAL_RESULT_${repo}
python report.py ${repo} chain_base xpu_test@baidu.com suijiaxin@baidu.com proxy-in.baidu.com >${ROOT_PATH}/FINAL_RESULT_${repo}
