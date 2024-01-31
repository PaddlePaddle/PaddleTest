#!/bin/bash

# 获取当前文件夹路径
cur_path=`pwd`
case_root=layercase_all
clas_case=layercase_all/Clas_cases
det_case=layercase_all/Det_cases
seg_case=layercase_all/Seg_cases
ocr_case=layercase_all/Ocr_cases

echo case_root is: ${case_root}

cd ${clas_case}

# 遍历当前文件夹下的子文件夹
for subdir in */; do
  echo ${subdir%/}
  tmp0=${subdir%/}
  tmp1=$(echo "$tmp0" | sed 's/ppcls\^configs\^ImageNet\^//g; s/_single_dy2st_train//g')
  tmp2=$(echo "$tmp1" | sed 's/\^/_/g; s/-/_/g')
  mv ${subdir} ${tmp2}

done

cd ${cur_path}

cd ${det_case}

# 遍历当前文件夹下的子文件夹
for subdir in */; do
  echo ${subdir%/}
  tmp0=${subdir%/}
  tmp1=$(echo "$tmp0" | sed 's/configs\^//g; s/_single_dy2st_train//g')
  tmp2=$(echo "$tmp1" | sed 's/\^/_/g; s/-/_/g')
  mv ${subdir} ${tmp2}

done

cd ${cur_path}

cd ${seg_case}

# 遍历当前文件夹下的子文件夹
for subdir in */; do
  echo ${subdir%/}
  tmp0=${subdir%/}
  tmp1=$(echo "$tmp0" | sed 's/configs\^//g; s/_single_dy2st_train//g')
  tmp2=$(echo "$tmp1" | sed 's/\^/_/g; s/-/_/g')
  mv ${subdir} ${tmp2}

done

cd ${cur_path}

cd ${ocr_case}

# 遍历当前文件夹下的子文件夹
for subdir in */; do
  echo ${subdir%/}
  tmp0=${subdir%/}
  tmp1=$(echo "$tmp0" | sed 's/configs\^//g; s/_single_dy2st_train//g')
  tmp2=$(echo "$tmp1" | sed 's/\^/_/g; s/-/_/g')
  mv ${subdir} ${tmp2}

done

cd ${cur_path}
