
export FLAGS_call_stack_level=2

# cd PaddleTest/distributed/CE_API && mkdir task
# cp -r run_gpu/. task && cp -r case/. task && cp -r test/. task
cd task
cases="test_dist_auto_api.py \
       test_dist_collective_communicator_api.py \
       test_dist_collective_stream_communicator_api.py \
       test_dist_rpc \
       test_dist_data_load.py  \
       test_dist_gloo_api.py \
       test_dist_env_training_api.py \
       test_dist_env_training_launch.py \
       test_dist_utils_api.py \
       test_dist_fleet_init.py \
       test_dist_fleet_distributedstrategy.py \
       test_dist_fleet_userdefinedrolemaker.py \
       test_dist_fleet_paddlecloudrolemaker.py \
       test_dist_fleet_utilbase.py \
       test_dist_fleet_utils.py \
       test_dist_fleet_worker.py \
       test_dist_fleet_dygraph.py \
       test_dist_fleet_dygraph_loss.py \
       test_dist_fleet_hybrid_parallel.py \
       test_dist_fleet_static_launch.py \
       test_dist_fleet_static_fleetrun.py \
       test_dist_MoE_api.py \
       "
for file in ${cases}
do
    echo ${file}
    python -m pytest  ${file} --alluredir=report
done
