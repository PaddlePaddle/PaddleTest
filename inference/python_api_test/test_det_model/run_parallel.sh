[[ -n $1 ]] && export CUDA_VISIBLE_DEVICES=$1
export FLAGS_call_stack_level=2
# V100 total:691.19s
cases="./test_fast_rcnn_mkldnn.py \
       ./test_fast_rcnn_gpu.py \
       ./test_fast_rcnn_trt_fp32.py \
       ./test_ppyolo_gpu.py \
       ./test_ppyolo_mkldnn.py \
       ./test_ppyolov2_mkldnn.py \
       ./test_solov2_gpu.py \
       ./test_solov2_mkldnn.py \
       ./test_yolov3_gpu.py \
       ./test_yolov3_mkldnn.py \
       ../test_nlp_model/test_bert_gpu.py \
       ../test_nlp_model/test_bert_mkldnn.py \
       ../test_nlp_model/test_bert_trt_fp32.py \
       ../test_nlp_model/test_ernie_gpu.py \
       ../test_nlp_model/test_ernie_mkldnn.py \
       ../test_nlp_model/test_ernie_trt_fp32.py \
       ../test_nlp_model/test_lac_gpu.py \
       ../test_nlp_model/test_lac_trt_fp32.py \
       ../test_nlp_model/test_lac_trt_fp16.py \
       ../test_nlp_model/test_AFQMC_base_trt_fp32.py \
       ../test_nlp_model/test_AFQMC_base_trt_fp16.py \
      "
bug=0

now=`date +'%Y-%m-%d %H:%M:%S'`
start_time=$(date --date="$now" +%s);
echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        python -m pytest -m server --disable-warnings -v ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

now=`date +'%Y-%m-%d %H:%M:%S'`
end_time=$(date --date="$now" +%s);
echo "card1 used time:"$((end_time-start_time))"s"

echo "total bugs: "${bug} >> result.txt
exit ${bug}
