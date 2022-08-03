#cd ./task/autograd

python3.7 -m pip install pytest
python3.7 -m pip install allure-pytest

case_dir_list=('base')
for case_dir in ${case_dir_list[@]}
do
python3.7 generate.py ${case_dir}
python3.7 test_${case_dir}.py ${case_dir}
done
