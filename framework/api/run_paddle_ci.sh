home=$PWD
python3.7 -m pip install pytest
python3.7 -m pip install scipy
export FLAGS_use_curand=1

case_dir_list=('fft' 'device' 'incubate' 'linalg' 'paddlebase' 'loss' 'nn' 'optimizer')
result_array=()
for case_dir in ${case_dir_list[@]}
do
rm -rf ${home}/$case_dir/result.txt
python3.7 multithreading_case.py $case_dir
result_array[${#result_array[@]}]=$?
wait;
done


# result
echo "=============== result ================="
for case_dir in ${case_dir_list[@]}
do
echo "[$case_dir cases result]"
cat ${home}/$case_dir/result.txt
done
for EXCODE in ${result_array[*]}
do
  if [ ${EXCODE} -ne 0 ]; then
    echo 'some case not success!'
    exit 8
  fi
done
