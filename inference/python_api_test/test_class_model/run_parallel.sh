[[ -n $1 ]] && export CUDA_VISIBLE_DEVICES=$1
export FLAGS_call_stack_level=2
# V100 total:690.25s
cases="./test_resnet50_gpu.py \
       ./test_resnet50_mkldnn.py \
       ./test_resnet50_trt_fp16.py \
       ./test_resnet50_trt_fp32.py \
       ./test_resnet50_slim.py \
       ./test_swin_transformer_gpu.py \
       ./test_swin_transformer_trt_fp32.py \
       ./test_tnt_small_gpu.py \
       ./test_vgg11_gpu.py \
       ./test_vgg11_mkldnn.py \
       ./test_ViT_base_patch16_224_trt_fp32.py \
      "
ignore=""
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
echo "card0 used time:"$((end_time-start_time))"s"

echo "total bugs: "${bug} >> result.txt
exit ${bug}
