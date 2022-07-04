#区分是否为重新训练的模型，没有../inference/"${model}的需要手动给出需要下载的预训练模型地址

function predict() {
    cd deploy


    case $category in
    ImageNet)
        size_tmp=`cat ${line} |grep image_shape|cut -d "," -f2|cut -d " " -f2` #获取train的shape保持和predict一致
        cd deploy
        sed -i 's/size: 224/size: '${size_tmp}'/g' configs/inference_cls.yaml #修改predict尺寸
        sed -i 's/resize_short: 256/resize_short: '${size_tmp}'/g' configs/inference_cls.yaml
        if [[ ${line} =~ 'ultra' ]];then
            python python/predict_cls.py -c configs/inference_cls_ch4.yaml  \
                -o Global.infer_imgs="./images"  \
                -o Global.batch_size=4 -o Global.inference_model_dir="../inference/"${model} \
                > ../$log_path/predict/${model}.log 2>&1
        else
            python python/predict_cls.py -c configs/inference_cls.yaml  \
                -o Global.infer_imgs="./images"  \
                -o Global.batch_size=4 \
                -o Global.inference_model_dir="../inference/"${model} \
                > ../$log_path/predict/${model}.log 2>&1
        fi
        if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ../$log_path/predict/${model}.log) -eq 0 ]];then
            echo -e "\033[33m multi_batch_size predict of ${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
            echo "predict_exit_code: 0.0" >> ../$log_path/predict/${model}.log
        else
            cat ../$log_path/predict/${model}.log
            echo -e "\033[31m multi_batch_size predict of ${model} failed!\033[0m"| tee -a ../$log_path/result.log
            echo "predict_exit_code: 1.0" >> ../$log_path/predict/${model}.log
        fi
        sed -i 's/size: '${size_tmp}'/size: 224/g' configs/inference_cls.yaml #改回predict尺寸
        sed -i 's/resize_short: '${size_tmp}'/resize_short: 256/g' configs/inference_cls.yaml
    ;;
    Cartoonface)
        python  python/predict_system.py -c configs/inference_cartoon.yaml \
            > ../$log_path/predict/${$category}.log 2>&1
        if [ $? -eq 0 ];then
            echo -e "\033[33m predict of ${category}_${model}  successfully!\033[0m"| tee -a ../$log_path/result.log
        else
            cat ../$log_path/predict/${$category}.log
        echo -e "\033[31m predict of ${category}_${model} failed!\033[0m"| tee -a ../$log_path/result.log
        fi
    ;;
    DeepHash)  #无
    
    ;;
    GeneralRecognition)
    
    ;;
    Logo)
    
    ;;
    metric_learning)
    
    ;;
    multi_scale)
    
    ;;
    practical_models)
    
    ;;
    Products)
    
    ;;
    PULC)
    
    ;;
    reid)
    
    ;;
    ResNet50_UReID_infer.yaml
    
    ;;
    slim)
    
    ;;
    StrategySearch)
    
    ;;
    Vehicle)
    
    ;;
    esac


    cd ..
}