#区分是否为重新训练的模型，没有../inference/"${model}的需要手动给出需要下载的预训练模型地址

function predict() {
    cd deploy

    if [[ -f "../inference/"${model}.pdparams ]];then
        local pretrained_model="../inference/"${model}
    else
        #直接执行predict，下载预训练模型
        # 需要提前下载预训练模型，怎么自动瞎下载?
        local pretrained_model=null

    case $category in
    ImageNet|slim|metric_learning)
        size_tmp=`cat ${line} |grep image_shape|cut -d "," -f2|cut -d " " -f2` #获取train的shape保持和predict一致
        cd deploy
        sed -i 's/size: 224/size: '${size_tmp}'/g' configs/inference_cls.yaml #修改predict尺寸
        sed -i 's/resize_short: 256/resize_short: '${size_tmp}'/g' configs/inference_cls.yaml
        if [[ ${line} =~ 'ultra' ]];then
            python python/predict_cls.py -c configs/inference_cls_ch4.yaml  \
                -o Global.infer_imgs="./images"  \
                -o Global.batch_size=4 -o Global.inference_model_dir=${pretrained_model} \
                > ../$log_path/predict/${model}.log 2>&1
        else
            python python/predict_cls.py -c configs/inference_cls.yaml  \
                -o Global.infer_imgs="./images"  \
                -o Global.batch_size=4 \
                -o Global.inference_model_dir=${pretrained_model} \
                > ../$log_path/predict/${model}.log 2>&1
        fi
        sed -i 's/size: '${size_tmp}'/size: 224/g' configs/inference_cls.yaml #改回predict尺寸
        sed -i 's/resize_short: '${size_tmp}'/resize_short: 256/g' configs/inference_cls.yaml
    ;;
    Cartoonface)
        python  python/predict_system.py -c configs/inference_cartoon.yaml \
            -o Global.inference_model_dir=${pretrained_model} \
            > ../$log_path/predict/${$category}.log 2>&1
    ;;
    DeepHash|GeneralRecognition)  #无
        python python/predict_rec.py -c configs/inference_rec.yaml \
            -o Global.rec_inference_model_dir=${pretrained_model} \
            > ../$log_path/predict/${$category}.log 2>&1
    ;;
    Logo)
        python  python/predict_system.py -c configs/inference_logo.yaml \
            -o Global.inference_model_dir=${pretrained_model} \
            > ../$log_path/predict/logo.log 2>&1
    ;;
    Products)
        python  python/predict_system.py -c configs/inference_product.yaml \
            -o Global.inference_model_dir=${pretrained_model} \
            > ../$log_path/predict/product.log 2>&1
    ;;
    PULC)
        # 抽取下一层 定义为策略
        python python/predict_cls.py -c configs/PULC/person_exists/inference_person_exists.yaml
        python python/predict_cls.py -c configs/PULC/${策略}/inference_/${策略}.yaml
            -o Global.inference_model_dir=${pretrained_model} \
            > ../$log_path/predict/${$category}.log 2>&1
    ;;
    reid)
        echo unspported
    ;;
    det) #没有训练过程，只有预测过程
        mkdir -p models
        # 下载通用检测 inference 模型并解压
        wget -P ./models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
        tar -xf ./models/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar -C ./models/
        python python/predict_det.py -c configs/inference_det.yaml
    ;;
    Vehicle)
        python  python/predict_system.py -c configs/inference_vehicle.yaml \
            -o Global.inference_model_dir=${pretrained_model} \
            > ../$log_path/predict/vehicle.log 2>&1
    ;;
    esac

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ../$log_path/predict/${model}.log) -eq 0 ]];then
        echo -e "\033[33m predict of ${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
        echo "predict_exit_code: 0.0" >> ../$log_path/predict/${model}.log
    else
        cat ../$log_path/predict/${model}.log
        echo -e "\033[31m predict of ${model} failed!\033[0m"| tee -a ../$log_path/result.log
        echo "predict_exit_code: 1.0" >> ../$log_path/predict/${model}.log
    fi

    cd ..
}
