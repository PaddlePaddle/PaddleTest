rm -rf test_*.py
cases=`find ../config/$1 -name "*.yaml" | sort`

echo $cases
ignoe=""
bug=0
#python${py_version} change_model.py
echo "============ failed cases =============" > result.txt
for file_dir in ${cases}
do
    name=`basename -s .yaml $file_dir`
    echo ${name}
    python${py_version} standardization.py -f ${file_dir}
    python${py_version} generate.py -f ${name} -a $1
    python${py_version} test_${name}.py
    if [ $? -ne 0 ]; then
        echo test_${name} >> result.txt
        bug=`expr ${bug} + 1`
    fi
    rm -rf *.npy
    rm -rf *.dat
done
echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}