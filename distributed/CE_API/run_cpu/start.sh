export FLAGS_call_stack_level=2

# cd PaddleTest/distributed/CE_API && mkdir task
# cp -r run_cpu/. task && cp -r case/. task && cp -r test/. task

cd task
cases="dist_data_inmemorydataset.py  \
       dist_data_queuedataset.py \
       dist_env_get_rank.py \
       dist_fleet_init.py \
       dist_fleet_init_collective.py \
       dist_fleet_init_role.py \
       dist_fleet_init_strategy.py \
       dist_fleet_distributedstrategy.py \
       dist_fleet_userdefinedrolemaker.py \
       dist_fleet_utils_localfs.py \
       dist_fleet_is_first_worker.py \
       dist_fleet_worker_index.py \
       dist_fleet_worker_num_cpu.py \
       dist_fleet_is_worker.py \
       dist_fleet_worker_endpoints_cpu.py \
       dist_fleet_server_num_cpu.py \
       dist_fleet_server_index.py \
       dist_fleet_server_endpoints_cpu.py \
       dist_fleet_is_server.py \
       dist_fleet_barrier_worker.py \
       dist_fleet_init_worker.py \
       dist_fleet_init_server.py \
       dist_fleet_server.py \
       dist_fleet_worker.py \
       dist_fleet_static_strategy.py \
       "
for file in ${cases}
do
    echo ${file}
    python -m pytest  ${file} --alluredir=report
done
