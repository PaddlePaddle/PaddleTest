#!/bin/bash

# 获取当前文件夹路径
cur_path=`pwd`
case_root=$1  # 设为layercase_all或layercase_unique等
clas_case=${case_root}/Clas_cases
det_case=${case_root}/Det_cases
seg_case=${case_root}/Seg_cases
ocr_case=${case_root}/Ocr_cases

echo case_root is: ${case_root}
cd ${case_root}

if [ -f "__init__.py" ]; then
    echo "delete root init ~~~"
    rm "__init__.py"
fi

# 遍历当前文件夹下的子文件夹
for subdir in */; do
  # 构建 import 语句
  import_statement="import ${case_root/\//.}.${subdir%/}"

  # 将 import 语句写入 __init__.py 文件
  echo "$import_statement" >> "__init__.py"
#   echo "$import_statement"

done

cd ${cur_path}

function generateRepoInit(){
repo_name=$1
root_dir=$2
cd ${repo_name}
if [ -f "__init__.py" ]; then
    echo "delete ${repo_name} init ~~~"
    rm "__init__.py"
fi

# 遍历当前文件夹下的子文件夹
for subdir in */; do
  # 构建 import 语句
  tmp0=${repo_name/\//.}.${subdir%/}
  import_statement="import ${tmp0}"

  # 将 import 语句写入 __init__.py 文件
  echo "$import_statement" >> "__init__.py"

  cd ${subdir}
  if [ -f "__init__.py" ]; then
    echo "delete ${subdir} init ~~~"
    rm "__init__.py"
  fi
  find ./ -type f -name "*.py" | while read file; do
    # echo file is: ${file}
    tmp1=$(echo "$file" | sed 's/.\///g; s/.py//g')
    import_statement="import ${tmp0}.${tmp1}"
    echo "$import_statement" >> "__init__.py"
  done
  cd ..

done

cd ${root_dir}

}

generateRepoInit ${clas_case} ${cur_path}
generateRepoInit ${det_case} ${cur_path}
generateRepoInit ${seg_case} ${cur_path}
generateRepoInit ${ocr_case} ${cur_path}
