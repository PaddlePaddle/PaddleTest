export FLAGS_call_stack_level=2
cases=`find . -name "test*.py" | sort`
ignore="test_deeplabv3_mkldnn.py \
        test_fcn_hrnetw18_mkldnn.py \
        test_pp_humanseg_lite_mkldnn.py \
        test_pp_liteseg_stdc1_mkldnn.py \
        test_unet_mkldnn.py
        "
bug=0

echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        python -m pytest -m jetson --disable-warnings -v ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

echo "total bugs: "${bug} >> result.txt
exit ${bug}
