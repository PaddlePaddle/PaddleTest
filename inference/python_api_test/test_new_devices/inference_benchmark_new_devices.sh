export FLAGS_cudnn_exhaustive_search=1
export FLAGS_allocator_strategy=auto_growth
export CUDA_MODULE_LOADING=LAZY
export FLAGS_conv_workspace_size_limit=32
export FLAGS_initial_cpu_memory_in_mb=0

backend_type_list=(MLU)
enable_trt_list=(false)
enable_gpu_list=(false)
enable_mkldnn_list=(false)
enable_gpu=false
enable_pir=false
enable_trt=false
precision=fp32
gpu_id=$1
batch_size_list=(1)
batch_size_list2=(1)
batch_size_list3=(1)
precision_list=(fp32)
subgraph_size=3
config_file=config.yaml
export BASEPATH=$(cd `dirname $0`; pwd)
export MODELPATH="$BASEPATH/Models"

run_benchmark(){
  if [ -f "$CONVERTPATH/done.txt" ];then
    model_count=$((${model_count}+1))
    echo "already been run!!"
    echo "===============model_count: $model_count==============="
    return 0
  fi

  for backend_type in ${backend_type_list[@]};do
    if [ "$backend_type" = "onnxruntime" ] || [ "$backend_type" = "openvino" ] ;then
      if [ ! -f "$model_dir/model.onnx" ];then
         echo "cont find ONNX model file. "
        continue
      fi
    fi
    model_file=""
    params_file=""
    for file in $(ls $model_dir)
      do
        if [ "${file##*.}"x = "pdmodel"x ];then
          model_file=$file
          echo "find model file: $model_file"
        fi

        if [ "${file##*.}"x = "pdiparams"x ];then
          params_file=$file
          echo "find param file: $params_file"
        fi
    done

    batch_size_var=${batch_size_list[@]}
    subgraph_size_var=${subgraph_size}

    for batch_size in ${batch_size_var[@]};do
      for enable_gpu in ${enable_gpu_list[@]};do
        if [ ${enable_gpu} = "true" ]; then
          python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --precision ${precision} --enable_gpu=${enable_gpu} --enable_pir=${enable_pir} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --subgraph_size=${subgraph_size_var} --paddle_model_file "$model_file" --paddle_params_file "$params_file" --return_result=true
        elif [ ${enable_gpu} = "false" ]; then
          python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --precision ${precision} --enable_gpu=${enable_gpu} --enable_pir=${enable_pir} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --subgraph_size=${subgraph_size_var} --paddle_model_file "$model_file" --paddle_params_file "$params_file" --return_result=true
        fi
      done
    done
  done

  touch $CONVERTPATH/done.txt
  model_count=$((${model_count}+1))
  echo "===============model_count: $model_count==============="
  # kill报错case遗留的僵尸进程，避免显存占用
  pkill -f benchmark.py
}

echo "============ Benchmark result =============" >> result.txt

model_count=0
for dir in $(ls $MODELPATH);do
  CONVERTPATH=$MODELPATH/$dir
  echo " >>>> Model path: $CONVERTPATH"
  export model_dir=$CONVERTPATH
  export dir_name=$dir
  run_benchmark
done
