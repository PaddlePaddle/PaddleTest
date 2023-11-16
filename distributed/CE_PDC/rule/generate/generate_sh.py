import yaml
import os
import sys

def generate_paddle_case(params):
    test_case_template = """
#!/usr/bin/env bash
# 执行位置在该模型套件的根目录
param="model_item={model_item_temp} "
param+="global_batch_size={base_batch_size_temp} "
param+="fp_item={fp_item_temp} "
param+="run_mode={run_mode_temp} "
param+="device_num={device_num_temp} "

# 以下为run_benchmark.sh的可选参数
param+="sequence_parallel=0 "
param+="pp_recompute_interval=1 "
param+="tensor_parallel_config=enable_mp_async_allreduce,enable_mp_skip_c_identity,enable_mp_fused_linear_param_grad_add "
param+="recompute_use_reentrant=true "

cd ./tests
# get data
bash {model_item_script_path_temp}/benchmark_common/prepare.sh {prepare_data_params_temp}
# run
bash -c "${param} bash {model_item_script_path_temp}/benchmark_common/run_benchmark.sh"
"""
    profiling_template = """
# run profiling
sleep 10;
export PROFILING=true
bash {model_item_script_path_temp}/benchmark_common/run_benchmark.sh ${{model_item}} ${{bs_item}} ${{fp_item}} ${{run_mode}} ${{device_num}} 2>&1;
unset PROFILING
"""

    # 分割参数值并为每个组合生成测试用例
    print(params)
    model_item = params['model_item']
    for base_batch_size in params['bs_item_list'].split('|'):
        for fp_item in params['fp_item_list'].split('|'):
            for run_mode in params['run_mode_list'].split('|'):
                for device_num in params['device_num_list'].split('|'):
                    test_case = test_case_template.format(
                        model_item_temp=model_item,
                        base_batch_size_temp=base_batch_size,
                        fp_item_temp=fp_item,
                        run_mode_temp=run_mode,
                        device_num_temp=device_num,
                        model_item_script_path_temp=params['model_item_script_path'],
                        prepare_data_params_temp=params['prepare_data_params'] if params['prepare_data_params'] != 'None' else ''
                    )
                    # if device_num == 'N1C1':
                    #     test_case += profiling_template.format(
                    #         model_item_script_path_temp=params['model_item_script_path']
                    #     )

                    # 创建目录
                    os.makedirs(os.path.join(params['model_item_script_path'], device_num), exist_ok=True)
                    with open(os.path.join(params['model_item_script_path'], device_num, f'{model_item}_bs{base_batch_size}_{fp_item}_{run_mode}.sh'), 'w') as f:
                        f.write(test_case)


def generate_pytorch_case(params):
    test_case_template = """
#!/usr/bin/env bash
# 执行位置在该模型套件的根目录
model_item={model_item_temp}
bs_item={base_batch_size_temp}
fp_item={fp_item_temp}
run_mode={run_mode_temp}
device_num={device_num_temp}
max_iter={max_iter_temp}
num_workers={num_workers_temp}
# get data
bash prepare.sh {prepare_data_params_temp}
# run
bash run_benchmark.sh ${{model_item}} ${{bs_item}} ${{fp_item}} ${{run_mode}} ${{device_num}} ${{max_iter}} ${{num_workers}} 2>&1;
"""

    # 分割参数值并为每个组合生成测试用例
    print(params)
    model_item = params['model_item']
    for base_batch_size in params['bs_item_list'].split('|'):
        for fp_item in params['fp_item_list'].split('|'):
            for run_mode in params['run_mode_list'].split('|'):
                for device_num in params['device_num_list'].split('|'):
                    test_case = test_case_template.format(
                        model_item_temp=model_item,
                        base_batch_size_temp=base_batch_size,
                        fp_item_temp=fp_item,
                        run_mode_temp=run_mode,
                        device_num_temp=device_num,
                        model_item_script_path_temp=params['model_item_script_path'],
                        prepare_data_params_temp=params['prepare_data_params'] if params['prepare_data_params'] != 'None' else ''
                    )
                    # 创建目录
                    os.makedirs(os.path.join(params['model_item_script_path'], model_item, device_num), exist_ok=True)
                    with open(os.path.join(params['model_item_script_path'], model_item, device_num, f'{model_item}_bs{base_batch_size}_{fp_item}_{run_mode}.sh'), 'w') as f:
                        f.write(test_case)


if __name__ == "__main__":
    # 获取传入的参数
    frame = sys.argv[1]
    config_path = sys.argv[2]
    #mode 读取 YAML 文件中的参数
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)
    if frame == 'paddle':
        generate_paddle_case(params)
    else:
        generate_pytorch_case(params)
    
