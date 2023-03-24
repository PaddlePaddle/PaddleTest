rm -rf test_*.py
export DDE_BACKEND=tensorflow.compat.v1
cases=`find ../../deepxde/examples/ -name "*.py" | sort `
ignore="dataset.py func_uncertainty.py func.py mf_dataset.py \
        mf_func.py antiderivative_aligned.py antiderivative_unaligned.py \
        Euler_beam.py Klein_Gordon.py \
        "
serial_bug=0
bug=0
echo "============ failed cases =============" > result.txt
for file in ${cases}
do
echo serial ${file} test
if [[ ${ignore} =~ ${file##*/} ]]; then
    echo "skip"
else
    python3.7 ${file}
    echo "test ${file} end" 
    if [ $? -ne 0 ]; then
        echo ${file} >> result.txt
        bug=`expr ${bug} + 1`
        serial_bug=`expr ${serial_bug} + 1`
    fi
fi
done
echo "serial bugs: "${serial_bug} >> result.txt