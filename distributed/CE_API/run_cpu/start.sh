export FLAGS_call_stack_level=2

# cd PaddleTest/distributed/CE_API && mkdir task
# cp -r run_cpu/. task && cp -r case/. task && cp -r test/. task

cd task
cases="dist_CountFilterEntry.py \
       dist_data_inmemorydataset.py  \
       dist_data_queuedataset.py \
       dist_DistAttr.py \
       dist_dtensor_from_fn.py \
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
       dist_fleet_qat_init.py \
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
       dist_load_state_dict.py \
       dist_fleet_static_strategy.py \
       dist_MultiSlotDataGenerator.py \
       dist_ParallelMode.py \
       dist_Partial.py \
       dist_Placement.py \
       dist_ProbabilityEntry.py \
       dist_ProcessMesh.py \
       dist_ReduceType.py \
       dist_Replicate.py \
       dist_save_state_dict.py \
       dist_shard_layer.py \
       dist_shard_optimizer.py \
       dist_shard_tensor.py \
       dist_Shard.py \
       dist_ShowClickEntry.py \
       dist_split.py \
       dist_Strategy.py \
       dist_utils_recompute.py \
       "
for file in ${cases}
do
    echo ${file}
    python -m pytest  ${file} --alluredir=report
done
