pip3.7 install pytest
export FLAGS_call_stack_level=2
cases=`find . -name "test*.py" | sort`
ignore="test_all.py \
test_allclose.py \
test_equal.py \
test_equal_all.py \
test_greater_equal.py \
test_greater_than.py \
test_less_equal.py \
test_less_than.py \
test_logical_and.py \
test_logical_not.py \
test_logical_or.py \
test_logical_xor.py \
test_not_equal.py \
test_is_empty.py \
test_is_tensor.py \
test_isfinite.py \
test_isinf.py \
test_isnan.py \

test_arange.py \
test_clip.py \
test_conj.py \
test_diagonal.py \
test_digamma.py \
test_divide.py \
test_empty_like.py \
test_erf.py \
test_expm1.py \
test_exp.py \
test_kron.py \
test_lgamma.py \
test_linspace.py \
test_log10.py \
test_log1p.py \
test_log2.py \
test_log.py \
test_logsumexp.py \
test_mean.py \
test_mm.py \
test_pow.py \
test_rsqrt.py \
test_scale.py \
test_sinh.py \
test_sin.py \
test_sqrt.py \
test_square.py \
test_stanh.py \
test_std.py \
test_sum.py \
test_tanh_.py \
test_tanh.py \
test_tan.py \
test_trace.py \
test_var.py \
test_floor_divide.py \
test_min.py \
test_prod.py"
bug=0

echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        python3.7 -m pytest ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

echo "total bugs: "${bug} >> result.txt
exit ${bug}
