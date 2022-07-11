cd ./task/competitor
python3.7 -m pip install pytest

case_dir_list=('base' 'nn')
for case_dir in ${case_dir_list[@]}
do
python3.7 yaml_executor.py $case_dir
done
