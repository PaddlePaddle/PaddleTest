export FLAGS_call_stack_level=2
cases="test_yolov3_gpu.py \
       test_yolov3_mkldnn.py \
       test_ppyolov2_trt_fp32.py \
       test_ppyolo_gpu.py \
       test_ppyolo_mkldnn.py \
       test_ppyolov2_gpu.py \
       test_ppyolov2_mkldnn.py \
       test_solov2_gpu.py \
       test_solov2_mkldnn.py"
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
