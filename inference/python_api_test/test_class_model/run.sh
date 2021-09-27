export FLAGS_call_stack_level=2
cases="./test_resnet50_gpu.py \
       ./test_resnet50_mkldnn.py \
       ./test_resnet50_trt_fp16.py \
       ./test_resnet50_trt_fp32.py \
       ./test_tnt_small_gpu.py \
       ./test_swin_transformer_trt_fp16.py \
       ./test_swin_transformer_mkldnn.py \
       ./test_resnet50_slim.py \
       ./test_tnt_small.py \
       ./test_pcpvt_base_gpu.py \
       ./test_pcpvt_base_mkldnn.py \
       ./test_pcpvt_base_trt_fp16.py \
       ./test_pcpvt_base_trt_fp32.py"
ignore=""
bug=0

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

echo "total bugs: "${bug} >> result.txt
exit ${bug}
