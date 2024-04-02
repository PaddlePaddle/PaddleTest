#!/usr/bin/env bash

unset GREP_OPTIONS
cur_path=`pwd`

while getopts ":P:p:t:g:L:h:o:H:" opt
do
    case $opt in
        P)
        echo "test paddle=$OPTARG"
        paddle=$OPTARG
        ;;
        p)
        echo "py version=$OPTARG"
        py_cmd=$OPTARG
        ;;
        t)
        echo "repo=$OPTARG"
        task=$OPTARG
        ;;
        g)
        echo "use gpu=$OPTARG"
        use_gpu=$OPTARG
        ;;
        L)
        echo "Linux sys=$OPTARG"
        linux_sys=$OPTARG
        ;;
        h)
        hub_config=$OPTARG
        ;;
        o)
        echo "pr_id=$OPTARG"
        pr_id=$OPTARG
        ;;
        H)
        http_proxy=$OPTARG
        ;;
        ?)
        echo "未知参数"
        usage
    esac
done

build_env(){
    root_path=`pwd`
    $py_cmd -m pip install --upgrade pip
    $py_cmd -m pip install pytest
    $py_cmd -m pip install ${paddle}

    git clone -b develop https://github.com/PaddlePaddle/PaddleHub.git
    cd PaddleHub
    git remote add upstream https://github.com/PaddlePaddle/PaddleHub
    $py_cmd -m pip install -r requirements.txt
    # $py_cmd setup.py install
    cd -

    $py_cmd -m pip install librosa

    $py_cmd -m pip install sentencepiece
    $py_cmd -m pip install pypinyin --upgrade
    $py_cmd -m pip install paddlex==1.3.11

    if [[ ${linux_sys} = "ubuntu" ]]; then
    apt-get install libsndfile1 -y
    fi
    if [[ ${linux_sys} = "centos" ]]; then
    yum install -y libsndfile
    fi

    export PYTHONPATH='.'

    unset https_proxy
    unset http_proxy
    git clone http://gitlab.baidu.com/chenxiaojie06/new_module_tool.git
    mv new_module_tool/* .

    echo start to auto_fill_content!!!!
    $py_cmd build_resource.py --pr_id ${pr_id}
    # sed -i "s/PaddleHub\/pull\/1816/PaddleHub\/pull\/${pr_id}/g" data/auto_fill_content.py
    # $py_cmd data/auto_fill_content.py

    cd PaddleHub
    $py_cmd setup.py install
    cd -

    hub config server==${hub_config}
    export http_proxy=${http_proxy}
    export https_proxy=${http_proxy}
}

ci(){
  # bash ci_test.sh -p python -t ci -g True
  # bash ci_test.sh -t ci -p python -g True -P ${compile_path} -L ubuntu -h ${hub_config}
  build_env

  root_path=`pwd`
  hub_excption=0
  hub_success=0
  hub_fail_list=
  hub_success_list=

  # $py_cmd data/auto_fill_content.py

  wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleHub/data/hub_CI_data.tar
  tar -xzf hub_CI_data.tar && mv hub_CI_data/* . && rm -rf hub_CI_data.tar

  for module in `ls Modules/files`
  do
  module_path=Modules/files/${module}
  model_path=Modules/tars/${module}

  cd Modules/tars
  tar -xzf ${module}*
  hub install ${module}

  if [ $? -ne 0 ];then
  echo ++++++++++++++++++++++${module} install Failed!!!++++++++++++++++++++++
  hub_excption=$(expr ${hub_excption} + 1)
  install_fail_list="${hub_fail_list} ${module}"
  continue
  else
  echo ++++++++++++++++++++++${module} install Success!!!++++++++++++++++++++++
  #hub_success=$(expr ${hub_success} + 1)
  install_success_list="${hub_success_list} ${module}"
  fi

  cd ${root_path}

  $py_cmd docbase.py --path ${module_path}/README.md --name ${module}
  sed -i "s/\/PATH\/TO\/IMAGE/doc_img.jpeg/g" test_${module}.py
  sed -i "s/\/PATH\/TO\/VIDEO/doc_video.mp4/g" test_${module}.py
  sed -i "s/\/PATH\/TO\/AUDIO/doc_audio.wav/g" test_${module}.py

  if [[ ${use_gpu} = "False" ]]; then
  sed -e "1i\paddle.set_device('cpu')" test_${module}.py
  sed -e "1i\import paddle" test_${module}.py
  sed -i "s/use_gpu=True/use_gpu=False/g" test_${module}.py
  else
  sed -e "1i\paddle.set_device('gpu')" test_${module}.py
  sed -e "1i\import paddle" test_${module}.py
  sed -i "s/use_gpu=False/use_gpu=True/g" test_${module}.py
  fi

  $py_cmd test_${module}.py

  if [ $? -ne 0 ];then
  echo ++++++++++++++++++++++${module} predict Failed!!!++++++++++++++++++++++
  hub_excption=$(expr ${hub_excption} + 1)
  hub_fail_list="${hub_fail_list} ${module}"
  continue
  else
  echo ++++++++++++++++++++++${module} predict Success!!!++++++++++++++++++++++
  hub_success=$(expr ${hub_success} + 1)
  hub_success_list="${hub_success_list} ${module}"
  fi


  done

  if [ ! -d "Modules/files" ]
  then
  echo Modules/files is empty!!!!!!
  hub_excption=$(expr ${hub_excption} + 1)
  fi

  echo ----------------------- num of bug modules is ${hub_excption} -----------------------
  echo install_fail_list: ${install_fail_list}
  echo fail modules are: ${hub_fail_list}
  echo install_success_list: ${install_success_list}
  echo success modules are: ${hub_success_list}
  exit ${hub_excption}
}

main(){
    case $task in
        (build_env)
            build_env
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
