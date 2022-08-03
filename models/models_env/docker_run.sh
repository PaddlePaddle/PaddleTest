ldconfig;
if [[ ${Python_env} == 'ln_way' ]];then
    # rm -rf /usr/bin/python2.7
    # rm -rf /usr/local/python2.7.15/bin/python
    # rm -rf /usr/local/bin/python
    # export PATH=/usr/local/bin/python:${PATH}
    case ${Python_version} in
    36)
    # ln -s /usr/local/bin/python3.6 /usr/local/bin/python
    mkdir run_env_py36;
    ln -s $(which python3.6) run_env_py36/python;
    ln -s $(which pip3.6) run_env_py36/pip;
    export PATH=$(pwd)/run_env_py36:${PATH};
    ;;
    37)
    # ln -s /usr/local/bin/python3.7 /usr/local/bin/python
    mkdir run_env_py37;
    ln -s $(which python3.7) run_env_py37/python;
    ln -s $(which pip3.7) run_env_py37/pip;
    export PATH=$(pwd)/run_env_py37:${PATH};
    ;;
    38)
    # ln -s /usr/local/bin/python3.8 /usr/local/bin/python
    mkdir run_env_py38;
    ln -s $(which python3.8) run_env_py38/python;
    ln -s $(which pip3.8) run_env_py38/pip;
    export PATH=$(pwd)/run_env_py38:${PATH};
    ;;
    39)
    # ln -s /usr/local/bin/python3.9 /usr/local/bin/python
    mkdir run_env_py39;
    ln -s $(which python3.9) run_env_py39/python;
    ln -s $(which pip3.9) run_env_py39/pip;
    export PATH=$(pwd)/run_env_py39:${PATH};
    ;;
    310)
    # ln -s /usr/local/bin/python3.10 /usr/local/bin/python
    mkdir run_env_py310;
    ln -s $(which python3.10) run_env_py310/python;
    ln -s $(which pip3.10) run_env_py310/pip;
    export PATH=$(pwd)/run_env_py310:${PATH};
    ;;
    esac
else [[ ${Python_env} == 'path_way' ]];then
    case ${Python_version} in
    36)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
    ;;
    37)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
    ;;
    38)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
    ;;
    39)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
    ;;
    310)
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.10.0/bin/:${PATH}
    ;;
    esac
fi

nvidia-smi;
python -c 'import sys; print(sys.version_info[:])';
git --version;
if [[ ${CE_version} == 'V2' ]];then
    bash main.sh --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type_id=${AGILE_PIPELINE_CONF_ID} --priority=${Priority_version} --compile_path=${Compile_version} --job_build_id=${AGILE_JOB_BUILD_ID};
else
    bash main.sh --task_type='model' --build_number=${AGILE_PIPELINE_BUILD_NUMBER} --project_name=${AGILE_MODULE_NAME} --task_name=${AGILE_PIPELINE_NAME}  --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type=${AGILE_PIPELINE_UUID} --owner='paddle' --priority=${Priority_version} --compile_path=${Compile_version} --agile_job_build_id=${AGILE_JOB_BUILD_ID};
fi