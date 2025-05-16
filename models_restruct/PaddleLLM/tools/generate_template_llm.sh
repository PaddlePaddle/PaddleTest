paddle_commit=`python -c "import paddle; print(paddle.version.commit)"`
paddlenlp_commit=`cd ./PaddleLLM && git rev-parse HEAD`
current_path=$PWD
root_path=`basename "$(dirname "$current_path")"`
ipipe_url=https://console.cloud.baidu-int.com/devops/ipipe/workspaces/${AGILE_WORKSPACE_ID}/pipeline-builds/${AGILE_PIPELINE_BUILD_ID}/stage-builds/${AGILE_STAGE_BUILD_ID}/view
allure_url=https://ipipe-report.baidu-int.com/bos/${root_path}/report/#behaviors
# kpi表格数据提取
log_dir="${current_path}/logs/PaddleLLM"
# 假设设置各模型的base值
declare -A base_values=(
    [grpo]=0.5
    [reinforce_plus_plus]=0.45
)
html_file="./utils/default_template_llm.html"

# 写入元信息
{
echo "<html><body>"
echo "<h3>任务信息</h3>"
echo "<p><b>iPipe 任务链接:</b> <a href='$ipipe_url' target='_blank'>$ipipe_url</a></p>"
echo "<p><b>Allure 报告链接:</b> <a href='$allure_url' target='_blank'>$allure_url</a></p>"
echo "<p><b>Paddle Commit:</b> $paddle_commit</p>"
echo "<p><b>PaddleNLP Commit:</b> $paddlenlp_commit</p>"
echo "<h3>评估结果</h3>"
echo "<table border='1'>"
echo "<tr><th>Model</th><th>Eval Accuracy</th><th>Base</th><th>Status</th></tr>"
} > "$html_file"

# 遍历所有 *_training.log 文件
find "$log_dir" -type f -name "*_training.log" | while read -r logfile; do
    # 提取模型名称（grpo、reinforce_plus_plus）
    model=$(basename "$logfile" | sed -E 's/.*_(grpo|reinforce_plus_plus)_training\.log/\1/')

    # 获取最后一个 eval_accuracy 的值
    acc=$(grep "eval_accuracy" "$logfile" | tail -n 1 | awk -F '=' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' | sed 's/[^0-9.]//g')

    # 如果没有拿到acc值，则设置为none
    if [[ -z "$acc" ]]; then
        acc="none"
    fi

    # 获取base值，默认为0.0
    base=${base_values[$model]:-0.0}

    # 判断是否异常（小于 base 就认为异常）
    status="正常"
    awk_res=$(awk -v a="$acc" -v b="$base" 'BEGIN {if (a < b) print "异常"; else print "正常"}')
    status="$awk_res"

    # 写入HTML表格
    echo "<tr><td>$model</td><td>$acc</td><td>$base</td><td>$status</td></tr>" >> "$html_file"
done

# HTML结束
echo "</table></body></html>" >> "$html_file"
