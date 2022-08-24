set -x
export http_proxy=${proxy}
export https_proxy=${proxy}
mkdir run_env_py37;
which python3.7
which pip3.7
ln -s $(which python3.7) run_env_py37/python;
ln -s $(which pip3.7) run_env_py37/pip;
export PATH=$(pwd)/run_env_py37:${PATH};
export PATH=/usr/local/bin:${PATH};
python -m pip install pip==20.2.4 --ignore-installed
python -m pip install pre-commit
python -m pip install clang-format
codestyle=on
git diff --numstat --diff-filter=AMR upstream/${branch} | tee file_list
for file_name in `cat file_list`;do
    pre-commit run --files $file_name
    if ! pre-commit run --files $file_name ; then
        codestyle=off
    fi
done 
if [ $codestyle == 'off' ];then
    echo "code format error"
    exit 1
fi
set +x
