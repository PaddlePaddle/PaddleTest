bash test_det_ce.sh 'release'
export build_type_id=${AGILE_PIPELINE_CONF_ID}
export build_id=${AGILE_PIPELINE_BUILD_ID}
export repo='Paddle'
path_temp=`echo ${paddle_compile} | awk -F 'paddlepaddle' '{print $1}'`
echo ${path_temp}
unset http_proxy
unset https_proxy
wget ${path_temp}description.txt
export commit_time=`grep commit_time description.txt | awk -F ':' '{print $2}'`
export branch=`grep branch description.txt | awk -F ' ' '{print $2}'`
export url=${url}
python push_result.py 
