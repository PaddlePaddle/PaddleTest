rm -rf test_*.py
cases=`find ../deepxde/examples/pinn_forward/ -name "*.py" | sort `
ignore=""
serial_bug=0
distributed_bug=0
bug=0
echo "============ failed cases =============" > result.txt
for file in ${cases}
do
echo serial ${file} test
if [[ ${ignore} =~ ${file##*/} ]]; then
    echo "skip"
else
    python3.7 standardization.py -f ${file}
    python3.7 ${file}
    if [ $? -ne 0 ]; then
        echo ${file} >> result.txt
        bug=`expr ${bug} + 1`
        serial_bug=`expr ${serial_bug} + 1`
    fi
fi
done
echo "serial bugs: "${serial_bug} >> result.txt