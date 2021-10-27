#!/usr/bin/env bash

cur_path=`pwd`

while getopts ":b:p:t:" opt
do
    case $opt in
        b)
        echo "test branch=$OPTARG"
        branch=$OPTARG
        ;;
        p)
        echo "py version=$OPTARG"
        py_cmd=$OPTARG
        ;;
        t)
        echo "task=$OPTARG"
        task=$OPTARG
        ;;
        ?)
        echo "未知参数"
        usage
    esac
done

image(){
   modules=`cat diff_image.log`
   excption=0
   for module in ${modules}
   do
   module_name=${module##*/}
   module_rdme=${module}/README.md
   echo ${module_rdme}
   #生成python测试代码
   $py_cmd docbase.py --path ${cur_path}/${module_rdme} --name ${module_name}
   #运行python测试代码
   sed -i "s/flower.jpg/doc_img.jpeg/g" test_${module_name}.py
   sed -i "s/\/PATH\/TO\/IMAGE/doc_img.jpeg/g" test_${module_name}.py
   $py_cmd test_${module_name}.py 2>&1 | tee -a predict_${module_name}.log
   excption=$(expr ${excption} + $?)
   done
   return ${excption}
}

audio(){
   modules=`cat diff_audio.log`
   for module in ${modules}
   do
   module_name=${module##*/}
   module_rdme=${module}/README.md
   echo ${module_rdme}
   #生成python测试代码
   $py_cmd docbase.py --path ${cur_path}/${module_rdme} --name ${module_name}
   #运行python测试代码
   $py_cmd test_${module_name}.py 2>&1 | tee -a predict_${module_name}.log
   done
}

video(){
   modules=`cat diff_video.log`
   for module in ${modules}
   do
   module_name=${module##*/}
   module_rdme=${module}/README.md
   echo ${module_rdme}
   #生成python测试代码
   $py_cmd docbase.py --path ${cur_path}/${module_rdme} --name ${module_name}
   #运行python测试代码
   $py_cmd test_${module_name}.py 2>&1 | tee -a predict_${module_name}.log
   done
}

text(){
   modules=`cat diff_text.log`
   excption=0
   for module in ${modules}
   do
   module_name=${module##*/}
   module_rdme=${module}/README.md
   echo ${module_rdme}
   #生成python测试代码
   $py_cmd docbase.py --path ${cur_path}/${module_rdme} --name ${module_name}
   #运行python测试代码
   $py_cmd test_${module_name}.py 2>&1 | tee -a predict_${module_name}.log
   excption=$(expr ${excption} + $?)
   done
   return ${excption}
}

ci(){
  text=True
  image=True
  audio=False
  video=False
  #提取包含modules的路径
  modules=`cat diff.log | grep "modules"`
  for module in ${modules}
  do
  echo ${module%/*} >> diff_modules_tmp.log
  done
  #删除重复行
  sort diff_modules_tmp.log | uniq >> diff_modules.log
  #记录不同类型module的路径
  modules=`cat diff_modules.log`
  for module in ${modules}
  do
  tmp_path=${module#modules/}
  module_type=${tmp_path%%/*}
  echo ${module} >> diff_${module_type}.log
  done

  text_error=0
  image_error=0
  audio_error=0
  video_error=0
  if [[ -f "diff_text.log" ]] && [[ "$text" = "True" ]]; then
    text
    text_error=$?
  fi

  if [[ -f "diff_image.log" ]] && [[ "$image" = "True" ]]; then
    image
    image_error=$?
  fi

  if [[ -f "diff_audio.log" ]] && [[ "$audio" = "True" ]]; then
    audio
    audio_error=$?
  fi

  if [[ -f "diff_video.log" ]] && [[ "$video" = "True" ]]; then
    video
    video_error=$?
  fi

  all_error=`expr ${text_error} + ${image_error} + ${audio_error} + ${video_error}`
  exit ${all_error}
}

ce(){
  text=True
  image=False
  audio=False
  video=False
}

main(){
    case $task in
        (ce)
            ce
            ;;
        (ci)
            ci
            ;;
        (*)
            echo "Error command"
            usage
            ;;
    esac
}

main
